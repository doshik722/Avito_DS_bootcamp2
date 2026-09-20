"""Финальный порядок: источники → top-300 → CatBoost + LightGBM → 50 item_id."""

import gc
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from .config import (
    ANSWER_TOP_K, BENCHMARK_CACHE, CATBOOST_MODEL, CATBOOST_TOP_K,
    CATBOOST_WEIGHT, COSINE_WEIGHT, LIGHTGBM_MODEL, LIGHTGBM_TREES,
    LIGHTGBM_WEIGHT, LOCAL_TOP_K, LOCAL_WEIGHT,
)
from .data import load_data
from .features import build_features
from .retrieval import (
    bm25_matrix, combine_text, e5_source, lexical_sources,
    local_top_k, parameter_source, source_candidates, top_positions,
)
from .signals import Filters, Geography


def _base_candidates(row, sources, source_scores, e5, parameters, filters):
    text_items, text_scores = source_candidates(row, sources, source_scores)
    e5_items = np.asarray(e5[0][row])
    valid = e5_items >= 0
    parts = [text_items, e5_items[valid]]
    code = filters.codes[row]
    if code >= 0:
        param_items = np.asarray(parameters[0][code])
        parts.append(param_items[param_items >= 0])
    candidates = np.unique(np.concatenate(parts))
    scores = np.zeros(len(candidates), dtype=np.float32)
    scores[np.searchsorted(candidates, text_items)] = text_scores
    e5_scores = np.asarray(e5[1][row], dtype=np.float32)[valid]
    if len(e5_scores) and e5_scores[0] > 0:
        scores[np.searchsorted(candidates, e5_items[valid])] += (
            COSINE_WEIGHT * e5_scores / e5_scores[0]
        )
    return candidates, scores


def _candidate_cache(n_queries, cache_dir, top_k):
    paths = [cache_dir / f"catboost_top{top_k}_{name}.npy"
             for name in ("items", "scores", "local_scores")]
    if all(path.exists() for path in paths):
        try:
            values = tuple(np.load(path, mmap_mode="r") for path in paths)
            if all(value.shape == (n_queries, top_k) for value in values):
                return values
        except (OSError, ValueError):
            pass
    return paths


def candidate_top300(data, geo, filters, sources, source_scores, e5, parameters,
                     cache_dir=BENCHMARK_CACHE, top_k=CATBOOST_TOP_K):
    """Включает географический BM25-пул; top_k по умолчанию равен 300."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = _candidate_cache(len(data.queries), cache_dir, top_k)
    if isinstance(existing[0], np.ndarray):
        print("CatBoost top-300: загружено")
        return existing

    queries, items = data.queries, data.items
    k = top_k
    chosen = np.full((len(queries), k), -1, dtype=np.int32)
    mixed_scores = np.zeros_like(chosen, dtype=np.float32)
    local_scores = np.zeros_like(chosen, dtype=np.float32)
    done = np.zeros(len(queries), dtype=bool)

    def store(row, local_items=None, local_values=None):
        base_items, base_score = _base_candidates(row, sources, source_scores,
                                                   e5, parameters, filters)
        candidates = (np.union1d(base_items, local_items)
                      if local_items is not None else base_items)
        current = np.zeros(len(candidates), dtype=np.float32)
        current[np.searchsorted(candidates, base_items)] = base_score
        current += geo.score(row, candidates)
        current += filters.bonus(row, candidates)
        if current.max() > 0:
            current /= current.max()
        local = np.zeros(len(candidates), dtype=np.float32)
        if local_items is not None and len(local_items) and local_values[0] > 0:
            local[np.searchsorted(candidates, local_items)] = local_values / local_values[0]
        mixed = (1 - LOCAL_WEIGHT) * current + LOCAL_WEIGHT * local
        top = top_positions(mixed, k)
        assert len(top) == k, "Корпус должен содержать не менее 300 кандидатов"
        chosen[row], mixed_scores[row], local_scores[row] = candidates[top], mixed[top], local[top]
        done[row] = True

    text = combine_text(items, ["item_title_raw", "item_description_raw"])
    vectorizer, item_matrix = bm25_matrix(text)
    query_matrix = vectorizer.transform(queries["search_query"].fillna("")).astype(np.float32)
    query_matrix.data.fill(1.0)
    for number, (rows, item_rows) in enumerate(geo.groups(), start=1):
        for row, local_items, values in local_top_k(query_matrix, item_matrix, rows,
                                                      item_rows, LOCAL_TOP_K):
            store(row, local_items, values)
        if number % 100 == 0:
            print(f"Локальный BM25: {number:,} локаций")
    for row in np.flatnonzero(~done):
        store(row)
    del vectorizer, item_matrix, query_matrix
    gc.collect()
    for path, value in zip(existing, (chosen, mixed_scores, local_scores)):
        np.save(path, value)
    return chosen, mixed_scores, local_scores


def _normalize_scores(values, n_queries):
    """Приводим скоры ранжировщика к 0–1 отдельно для каждого запроса."""
    values = np.asarray(values, dtype=np.float32).reshape(n_queries, CATBOOST_TOP_K)
    return ((values - values.min(axis=1, keepdims=True))
            / np.maximum(np.ptp(values, axis=1, keepdims=True), 1e-8))


def predict(data_dir=None):
    """Возвращает таблицу ответа; файлы на диск пишет вызывающий скрипт."""
    if not CATBOOST_MODEL.exists():
        from train import train_model
        print("Файл CatBoost отсутствует; восстанавливаем модель из train.parquet")
        train_model()
    if not LIGHTGBM_MODEL.exists():
        from train import train_lightgbm_model
        print("Файл LightGBM отсутствует; восстанавливаем модель из train.parquet")
        train_lightgbm_model()
    data = load_data() if data_dir is None else load_data(data_dir)
    queries, items = data.queries, data.items
    sources, source_scores = lexical_sources(queries, items)
    filters = Filters.from_data(queries, items)
    parameters = parameter_source(items, filters.texts)
    e5 = e5_source(queries, items)
    geo = Geography.from_data(data)
    candidates, scores, local_scores = candidate_top300(
        data, geo, filters, sources, source_scores, e5, parameters,
    )
    features = build_features(data, geo, filters, candidates, scores, local_scores,
                              sources, source_scores, *e5, *parameters)
    model = CatBoostRanker()
    model.load_model(str(CATBOOST_MODEL))
    assert features.columns.tolist() == model.feature_names_, "Порядок признаков изменился"
    catboost_scores = _normalize_scores(model.predict(features), len(queries))
    catboost_mix = (1 - CATBOOST_WEIGHT) * scores + CATBOOST_WEIGHT * catboost_scores

    # ZIP помещается в обычный Git; временный текстовый файл нужен LightGBM для загрузки.
    with TemporaryDirectory(dir=BENCHMARK_CACHE) as temporary, ZipFile(LIGHTGBM_MODEL) as archive:
        model_file = archive.extract("lightgbm_all4500_top300.txt", path=temporary)
        lightgbm_model = lgb.Booster(model_file=model_file)
    assert features.columns.tolist() == lightgbm_model.feature_name(), "Порядок признаков LightGBM изменился"
    lightgbm_scores = _normalize_scores(
        lightgbm_model.predict(features, num_iteration=LIGHTGBM_TREES), len(queries)
    )
    final_scores = (1 - LIGHTGBM_WEIGHT) * catboost_mix + LIGHTGBM_WEIGHT * lightgbm_scores
    top = np.stack([candidates[row, top_positions(final_scores[row], ANSWER_TOP_K)]
                    for row in range(len(queries))])
    item_ids = items["item_id"].astype(str).to_numpy()
    answer = pd.DataFrame({
        "query_id": queries["query_id"].astype(str).to_numpy(),
        "answer": [" ".join(item_ids[row]) for row in top],
    })
    validate_answer(answer, queries, items)
    return answer


def validate_answer(answer, queries, items):
    """Ошибку формата ловим до отправки на платформу."""
    ids = answer["answer"].str.split()
    known = set(items["item_id"].astype(str))
    assert answer.columns.tolist() == ["query_id", "answer"]
    assert len(answer) == len(queries) and answer["query_id"].is_unique
    assert set(answer["query_id"]) == set(queries["query_id"].astype(str))
    assert answer["query_id"].str.len().eq(16).all()
    assert ids.map(lambda values: len(values) == ANSWER_TOP_K
                   and len(set(values)) == ANSWER_TOP_K
                   and set(values).issubset(known)).all()
