"""Восстановление данных для сравнения ранжировщиков после перезапуска ядра."""

import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRanker

from .config import ARTIFACTS, CATBOOST_TOP_K, CATBOOST_WEIGHT, DATA, SOURCE_NAMES
from .data import Datasets, prepare_text
from .features import build_features
from .pipeline import candidate_top300
from .retrieval import parameter_source, top_positions
from .signals import Filters, Geography


class SavedRows:
    """Читает из большого сохранённого массива только нужный запрос."""

    def __init__(self, path, selected_rows):
        self.array = np.load(path, mmap_mode="r")
        self.selected_rows = selected_rows

    def __getitem__(self, row):
        return self.array[self.selected_rows[row]]


def restore_ranking_validation(top_k=CATBOOST_TOP_K, validation_only=False, train_only=False):
    """Восстанавливает кандидатов и признаки без нового расчёта E5."""
    assert not (validation_only and train_only)
    # Восстанавливаем обучающий корпус. Benchmark-файлы читаем лишь как схему:
    # загружать их целиком для валидации не нужно.
    train = prepare_text(pd.read_parquet(DATA / "train.parquet", dtype_backend="pyarrow"))
    query_columns = [
        name for name in pq.read_schema(DATA / "benchmark_queries.parquet").names
        if name != "query_id"
    ]
    item_columns = pq.read_schema(DATA / "benchmark_items.parquet").names
    train_queries = train[query_columns].drop_duplicates().reset_index(drop=True)
    train_queries.insert(0, "train_query_id", np.arange(len(train_queries)))
    train_items = train[item_columns].drop_duplicates("item_id").reset_index(drop=True)
    relevance = (train[query_columns + ["item_id"]]
                 .merge(train_queries, on=query_columns, how="left")
                 [["train_query_id", "item_id"]].drop_duplicates().reset_index(drop=True))
    del train
    gc.collect()

    # Строго тот же порядок 4 500 запросов и объявлений, что был у CatBoost.
    valid_texts = (train_queries["search_query"].drop_duplicates()
                   .sample(frac=0.2, random_state=42))
    queries_valid = train_queries.loc[train_queries["search_query"].isin(valid_texts)]
    saved = ARTIFACTS / "retrieval" / "validation_leaders"
    saved_query_ids = np.load(saved / "query_ids.npy")
    assert np.array_equal(saved_query_ids, queries_valid["train_query_id"].to_numpy())
    assert np.array_equal(np.load(saved / "item_ids.npy"),
                          train_items["item_id"].astype(str).to_numpy())
    query_table = queries_valid.set_index("train_query_id").reindex(saved_query_ids)
    rows = (query_table.reset_index(drop=True).drop_duplicates("search_query")
            .index.to_numpy(np.int32))
    np.random.default_rng(42).shuffle(rows)
    n_train, n_valid = 3000, 1500
    rows = rows[:n_train + n_valid]
    assert len(rows) == n_train + n_valid
    valid_rows = rows[n_train:]
    if validation_only:
        rows = valid_rows
    elif train_only:
        rows = rows[:n_train]
    selected = query_table.iloc[rows].reset_index().rename(
        columns={"train_query_id": "query_id"})
    selected_ids = selected["query_id"].to_numpy()

    # Глобальные BM25/TF-IDF и E5 уже сохранены; mmap не копирует гигабайтные
    # массивы в оперативную память. Пересчитываем только локальные сигналы.
    sources = {name: SavedRows(saved / f"{name}_top5000_indices.npy", rows)
               for name in SOURCE_NAMES}
    source_scores = {name: SavedRows(saved / f"{name}_top5000_scores.npy", rows)
                     for name in SOURCE_NAMES}
    e5 = (SavedRows(saved / "multilingual_e5_title_description_top5000_indices.npy", rows),
          SavedRows(saved / "multilingual_e5_title_description_top5000_scores.npy", rows))
    relevance_train = relevance.loc[
        ~relevance["train_query_id"].isin(queries_valid["train_query_id"])]
    data = Datasets(selected, train_items, train_queries, train_items, relevance_train)
    # Для top-300 оставляем прежний кэш; top-500 не перезапишет его файлы.
    if train_only:
        cache_name = f"top{top_k}_train"
    else:
        cache_name = ("logreg_validation" if top_k == 300 and not validation_only
                      else f"top{top_k}_validation")
    cache = ARTIFACTS / "retrieval" / cache_name
    cache.mkdir(parents=True, exist_ok=True)
    order_path = cache / "query_ids.npy"
    if order_path.exists():
        assert np.array_equal(np.load(order_path), selected_ids), "Изменился порядок запросов"
    else:
        np.save(order_path, selected_ids)

    filters = Filters.from_data(selected, train_items)
    parameters = parameter_source(train_items, filters.texts, cache_dir=cache)
    geo = Geography.from_data(data)
    items, scores, local_scores = candidate_top300(
        data, geo, filters, sources, source_scores, e5, parameters,
        cache_dir=cache, top_k=top_k)
    features_path = cache / "features.parquet"
    if features_path.exists():
        features = pd.read_parquet(features_path)
    else:
        features = build_features(data, geo, filters, items, scores, local_scores,
                                  sources, source_scores, *e5, *parameters, top_k=top_k)
        features.to_parquet(features_path, index=False)

    # Метки восстанавливаются из выбранных в train пар, а не из предсказаний.
    item_positions = pd.Index(train_items["item_id"].astype(str))
    pairs = relevance.loc[relevance["train_query_id"].isin(selected_ids)].copy()
    pairs["item_index"] = item_positions.get_indexer(pairs["item_id"].astype(str))
    assert pairs["item_index"].ge(0).all()
    relevant = pairs.groupby("train_query_id")["item_index"].agg(set).to_dict()
    labels = np.concatenate([
        np.isin(items[row], list(relevant[query_id]))
        for row, query_id in enumerate(selected_ids)
    ]).astype(np.int8)
    train_groups = (np.empty(0, dtype=np.int32) if validation_only else np.flatnonzero(
        labels.reshape(len(selected_ids), top_k)[:n_train].any(axis=1)))

    def recall_for_item_indices(row, item_indices):
        actual = relevant[saved_query_ids[row]]
        return sum(int(item) in actual for item in item_indices) / len(actual)

    if train_only:
        print(f"top-{top_k}: {len(selected_ids):,} обучающих запросов; группы с релевантными: {len(train_groups):,}")
        return {"cb_features": features, "cb_labels": labels,
                "cb_extra_train_groups": train_groups}

    # Для top-300 сверяем прежний Recall; top-500 — новый эксперимент.
    model = CatBoostRanker()
    model.load_model(str(ARTIFACTS / "models" / "catboost_extra_top300_valid.cbm"))
    assert features.columns.tolist() == model.feature_names_
    offset = 0 if validation_only else n_train
    model_scores = model.predict(features.iloc[offset * top_k:]).reshape(
        n_valid, top_k).astype(np.float32)
    model_scores = ((model_scores - model_scores.min(axis=1, keepdims=True))
                    / np.maximum(np.ptp(model_scores, axis=1, keepdims=True), 1e-8))
    blended = (1 - CATBOOST_WEIGHT) * scores[offset:] + CATBOOST_WEIGHT * model_scores
    restored_recall = np.mean([
        recall_for_item_indices(row, items[offset + pos][top_positions(blended[pos], 50)])
        for pos, row in enumerate(valid_rows)
    ])
    print(f"top-{top_k}: {len(selected_ids):,} запросов; Recall@50 = {restored_recall:.4f}")
    if top_k == 300 and not validation_only:
        assert abs(restored_recall - 0.8320) < 0.002, "Прежний Recall не совпал"

    return {
        "CB_MAX_K": top_k,
        "CB_TRAIN_QUERIES": offset,
        "CB_VALID_QUERIES": n_valid,
        "cb_features": features,
        "cb_items": items,
        "current_scores": blended,
        "cb_labels": labels,
        "cb_extra_train_groups": train_groups,
        "model_valid_rows": valid_rows,
        "recall_for_item_indices": recall_for_item_indices,
    }
