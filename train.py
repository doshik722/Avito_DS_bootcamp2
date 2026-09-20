"""Запасной путь: восстановить ранжировщики, если готовых моделей нет.

Если сохранённые модели есть, предсказание просто загружает их. Обучение здесь —
запасной путь для воспроизводимости, существенно более долгий, чем загрузка.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRanker, Pool
from zipfile import ZIP_DEFLATED, ZipFile

from avito_candidates.config import (
    ARTIFACTS, CATBOOST_MODEL, CATBOOST_TOP_K, DATA,
    LIGHTGBM_MODEL, LIGHTGBM_TREES,
)
from avito_candidates.data import Datasets, load_data, normalize_text
from avito_candidates.features import CATEGORICAL, build_features
from avito_candidates.pipeline import candidate_top300
from avito_candidates.retrieval import e5_source, lexical_sources, parameter_source
from avito_candidates.signals import Filters, Geography


def _training_queries(data):
    """То же разделение по текстам и те же 4 500 запросов, что в ноутбуке."""
    groups = data.train_queries["search_query"]
    validation_texts = groups.drop_duplicates().sample(frac=0.2, random_state=42)
    valid = data.train_queries.loc[groups.isin(validation_texts)].copy()
    rows = valid.reset_index(drop=True).drop_duplicates("search_query").index.to_numpy(np.int32)
    np.random.default_rng(42).shuffle(rows)
    selected = valid.iloc[rows[:4500]].reset_index(drop=True)
    assert len(selected) == 4500
    return selected.rename(columns={"train_query_id": "query_id"})


def _training_e5(bundle, cache_dir):
    """Берём сохранённые validation top-5000; если их нет, считаем E5 локально."""
    old = ARTIFACTS / "retrieval" / "validation_leaders"
    ids_path = old / "query_ids.npy"
    indices_path = old / "multilingual_e5_title_description_top5000_indices.npy"
    scores_path = old / "multilingual_e5_title_description_top5000_scores.npy"
    if all(path.exists() for path in (ids_path, indices_path, scores_path)):
        positions = pd.Index(np.load(ids_path)).get_indexer(bundle.queries["query_id"])
        if (positions >= 0).all():
            return (np.asarray(np.load(indices_path, mmap_mode="r")[positions]),
                    np.asarray(np.load(scores_path, mmap_mode="r")[positions]))

    raw = pd.read_parquet(DATA / "train.parquet",
                          columns=["item_id", "item_title_raw", "item_description_raw", "search_query"])
    raw_items = raw[["item_id", "item_title_raw", "item_description_raw"]].drop_duplicates("item_id")
    raw_items = raw_items.assign(item_id=raw_items["item_id"].astype(str))
    raw_items = raw_items.set_index("item_id").reindex(bundle.items["item_id"].astype(str).to_numpy())
    item_texts = ("passage: " + raw_items["item_title_raw"].fillna("").astype(str).str.strip()
                  + ". " + raw_items["item_description_raw"].fillna("").astype(str).str.strip()).to_numpy(str)
    raw_queries = pd.DataFrame({
        "raw": raw["search_query"].fillna("").astype(str),
        "normalized": normalize_text(raw["search_query"]).astype(str),
    }).drop_duplicates("normalized").set_index("normalized")["raw"]
    original = bundle.queries["search_query"].fillna("").astype(str).map(raw_queries)
    original = original.fillna(bundle.queries["search_query"].fillna("").astype(str))
    query_texts = ("query: " + original.str.strip()).to_numpy(str)
    return e5_source(bundle.queries, bundle.items, cache_dir=cache_dir / "e5",
                     item_texts=item_texts, query_texts=query_texts)


def _training_features():
    """Одни и те же 4 500 групп и 44 признака для двух финальных моделей."""
    data = load_data()
    queries = _training_queries(data)
    cache = ARTIFACTS / "retrieval" / "training_final"
    features_path, labels_path = cache / "features.parquet", cache / "labels.npy"
    query_ids_path, item_ids_path = cache / "query_ids.npy", cache / "item_ids.npy"
    query_ids = queries["query_id"].to_numpy()
    item_ids = data.train_items["item_id"].astype(str).to_numpy(dtype=str)
    if all(path.exists() for path in (features_path, labels_path, query_ids_path, item_ids_path)):
        if (np.array_equal(np.load(query_ids_path), query_ids)
                and np.array_equal(np.load(item_ids_path), item_ids)):
            print("Обучающие признаки: загружены")
            return pd.read_parquet(features_path), np.load(labels_path)

    # Гео-статистику, как в валидационном эксперименте, строим без кликов
    # отложенных 20% текстов запроса; сами объявления остаются в корпусе.
    heldout_texts = (data.train_queries["search_query"].drop_duplicates()
                     .sample(frac=0.2, random_state=42))
    heldout_ids = data.train_queries.loc[
        data.train_queries["search_query"].isin(heldout_texts), "train_query_id"
    ]
    geo_relevance = data.relevance.loc[
        ~data.relevance["train_query_id"].isin(heldout_ids)
    ]
    bundle = Datasets(queries, data.train_items, data.train_queries,
                      data.train_items, geo_relevance)
    sources, source_scores = lexical_sources(queries, bundle.items, cache_dir=cache)
    filters = Filters.from_data(queries, bundle.items)
    parameters = parameter_source(bundle.items, filters.texts, cache_dir=cache)
    e5 = _training_e5(bundle, cache)
    geo = Geography.from_data(bundle)
    candidates, scores, local_scores = candidate_top300(
        bundle, geo, filters, sources, source_scores, e5, parameters, cache_dir=cache,
    )
    features = build_features(bundle, geo, filters, candidates, scores, local_scores,
                              sources, source_scores, *e5, *parameters)
    item_rows = pd.Index(bundle.items["item_id"].astype(str))
    relevant = data.relevance.groupby("train_query_id")["item_id"].agg(list)
    labels = np.concatenate([
        np.isin(candidates[row], item_rows.get_indexer(relevant.loc[query_id]))
        for row, query_id in enumerate(queries["query_id"])
    ]).astype(np.int8)
    cache.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_path, index=False)
    np.save(labels_path, labels)
    np.save(query_ids_path, query_ids)
    np.save(item_ids_path, item_ids)
    return features, labels


def train_model():
    features, labels = _training_features()
    pool = Pool(features, label=labels,
                group_id=np.repeat(np.arange(len(labels) // CATBOOST_TOP_K, dtype=np.int32), CATBOOST_TOP_K),
                cat_features=CATEGORICAL)
    model = CatBoostRanker(loss_function="YetiRank", iterations=294, depth=6,
                           learning_rate=0.08, l2_leaf_reg=5, random_seed=42,
                           thread_count=4, allow_writing_files=False, verbose=50)
    model.fit(pool)
    CATBOOST_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(CATBOOST_MODEL))
    print(f"Сохранена модель: {CATBOOST_MODEL}")


def train_lightgbm_model():
    """Запасной путь, если в репозитории нет сжатой готовой модели."""
    features, labels = _training_features()
    groups = np.flatnonzero(labels.reshape(-1, CATBOOST_TOP_K).any(axis=1))
    rows = (groups[:, None] * CATBOOST_TOP_K + np.arange(CATBOOST_TOP_K)).ravel()
    model = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=LIGHTGBM_TREES, learning_rate=0.05,
        num_leaves=31, min_child_samples=100, reg_lambda=5.0,
        n_jobs=4, random_state=42, verbosity=-1,
    )
    model.fit(features.iloc[rows], labels[rows],
              group=np.full(len(groups), CATBOOST_TOP_K),
              categorical_feature=CATEGORICAL)
    raw = ARTIFACTS / "retrieval" / "training_final" / "lightgbm_all4500_top300.txt"
    model.booster_.save_model(str(raw))
    LIGHTGBM_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(LIGHTGBM_MODEL, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(raw, arcname="lightgbm_all4500_top300.txt")
    print(f"Сохранена модель: {LIGHTGBM_MODEL}")


if __name__ == "__main__":
    train_model()
    train_lightgbm_model()
