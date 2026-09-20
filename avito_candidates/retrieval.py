"""Только поисковые источники, входящие в финальный пайплайн."""

import gc
import json
import os

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .config import (
    BENCHMARK_CACHE, DATA, E5_CACHE, E5_MODEL, E5_TOP_K,
    SOURCE_NAMES, SOURCE_WEIGHTS, TEXT_TOP_K,
)


def top_positions(scores, k):
    """Индексы k лучших элементов: порядок важен для признаков CatBoost."""
    k = min(k, len(scores))
    if not k:
        return np.empty(0, dtype=np.int32)
    positions = np.arange(k) if k == len(scores) else np.argpartition(scores, -k)[-k:]
    return positions[np.argsort(scores[positions])[::-1]]


def combine_text(frame, columns):
    text = frame[columns[0]].fillna("")
    for column in columns[1:]:
        text = text.str.cat(frame[column].fillna(""), sep=" ")
    return text


def bm25_matrix(text, k1=1.5, b=0.75):
    """BM25-веса в разреженной матрице; без квадратичной матрицы запрос×корпус."""
    vectorizer = CountVectorizer(dtype=np.float32)
    counts = vectorizer.fit_transform(text)
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(1 + (counts.shape[0] - df + 0.5) / (df + 0.5))
    norm = k1 * (1 - b + b * lengths / lengths.mean())
    matrix = counts.copy()
    matrix.data *= (k1 + 1) / (matrix.data + np.repeat(norm, np.diff(matrix.indptr)))
    matrix.data *= idf[matrix.indices]
    return vectorizer, matrix.tocsr()


def retrieve(text, queries, method, k, batch_size=128):
    if method == "bm25":
        vectorizer, items = bm25_matrix(text)
        query_matrix = vectorizer.transform(queries).astype(np.float32)
        query_matrix.data.fill(1.0)
    else:
        vectorizer = (TfidfVectorizer(dtype=np.float32) if method == "word"
                      else TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                           min_df=3, max_features=100_000, dtype=np.float32))
        items = vectorizer.fit_transform(text)
        query_matrix = vectorizer.transform(queries)
    indices = np.full((len(queries), k), -1, dtype=np.int32)
    scores = np.full((len(queries), k), -np.inf, dtype=np.float32)
    for start in range(0, len(queries), batch_size):
        batch = (query_matrix[start:start + batch_size] @ items.T).tocsr()
        for offset in range(batch.shape[0]):
            row = batch.getrow(offset)
            top = top_positions(row.data, k)
            indices[start + offset, :len(top)] = row.indices[top]
            scores[start + offset, :len(top)] = row.data[top]
    return indices, scores


def load_or_retrieve(name, text, queries, method, k=TEXT_TOP_K, batch_size=128,
                     cache_dir=BENCHMARK_CACHE):
    """Сохранённые top-k загружаются; отсутствующие пересчитываются один раз."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_dir / f"{name}_{part}.npy" for part in ("indices", "scores")]
    if all(path.exists() for path in paths):
        try:
            arrays = tuple(np.load(path, mmap_mode="r") for path in paths)
            if all(array.shape == (len(queries), k) for array in arrays):
                print(f"{name}: загружено")
                return arrays
        except (OSError, ValueError):
            pass
    result = retrieve(text, queries, method, k, batch_size)
    for path, array in zip(paths, result):
        np.save(path, array)
    print(f"{name}: рассчитано и сохранено")
    return result


def lexical_sources(queries, items, cache_dir=BENCHMARK_CACHE):
    title = items["item_title_raw"].fillna("")
    title_description = combine_text(items, ["item_title_raw", "item_description_raw"])
    configs = (("bm25", title_description, 128), ("char", title, 64),
               ("bm25", title, 128), ("word", title, 128))
    found = [load_or_retrieve(name, text, queries["search_query"].fillna(""),
                              method, batch_size=batch, cache_dir=cache_dir)
             for name, (method, text, batch) in zip(SOURCE_NAMES, configs)]
    return ({name: pair[0] for name, pair in zip(SOURCE_NAMES, found)},
            {name: pair[1] for name, pair in zip(SOURCE_NAMES, found)})


def parameter_source(items, filters, cache_dir=BENCHMARK_CACHE):
    return load_or_retrieve("bm25_parameters", items["item_infm_params_text"].fillna(""),
                            filters, "bm25", batch_size=32, cache_dir=cache_dir)


def source_candidates(row, indices_by_name, scores_by_name):
    parts = [np.asarray(indices_by_name[name][row]) for name in SOURCE_NAMES]
    candidates = np.unique(np.concatenate([part[part >= 0] for part in parts]))
    scores = np.zeros(len(candidates), dtype=np.float32)
    for name, weight, part in zip(SOURCE_NAMES, SOURCE_WEIGHTS, parts):
        valid = part >= 0
        if not weight or not valid.any():
            continue
        values = np.asarray(scores_by_name[name][row][valid], dtype=np.float32)
        if values.max() > 0:
            scores[np.searchsorted(candidates, part[valid])] += weight * values / values.max()
    return candidates, scores


def local_top_k(query_matrix, item_matrix, rows, item_indices, k):
    """Тот же BM25, но только в географическом подкорпусе."""
    local = item_matrix[item_indices].T.tocsr()
    for start in range(0, len(rows), 64):
        batch = (query_matrix[rows[start:start + 64]] @ local).tocsr()
        for offset, row in enumerate(rows[start:start + 64]):
            scores = batch.getrow(offset)
            top = top_positions(scores.data, k)
            yield row, item_indices[scores.indices[top]], scores.data[top]


def _embeddings(texts, path, model, batch_size, chunk=2048):
    """Долгий расчёт продолжается с последнего полностью сохранённого блока."""
    dimension = model.get_embedding_dimension()
    complete = path.with_name(path.stem + "_complete.json")
    progress = path.with_name(path.stem + "_progress.json")
    meta = {"rows": len(texts), "dimension": dimension, "max_seq_length": 128}
    if path.exists() and complete.exists():
        saved = json.loads(complete.read_text())
        if all(saved.get(key) == value for key, value in meta.items()):
            vectors = np.load(path, mmap_mode="r")
            if vectors.shape == (len(texts), dimension):
                return vectors
    start = 0
    if path.exists() and progress.exists():
        state = json.loads(progress.read_text())
        if all(state.get(key) == value for key, value in meta.items()):
            start = state["next_row"]
            result = np.load(path, mmap_mode="r+")
    if not start:
        result = open_memmap(path, mode="w+", dtype=np.float32, shape=(len(texts), dimension))
    model.max_seq_length = 128
    for first in range(start, len(texts), chunk):
        stop = min(first + chunk, len(texts))
        result[first:stop] = model.encode(texts[first:stop].tolist(), batch_size=batch_size,
                                          convert_to_numpy=True, normalize_embeddings=True)
        result.flush()
        progress.write_text(json.dumps({**meta, "next_row": stop}))
        print(f"Эмбеддинги: {stop:,}/{len(texts):,}")
    complete.write_text(json.dumps(meta))
    progress.unlink(missing_ok=True)
    return np.load(path, mmap_mode="r")


def e5_source(queries, items, data_dir=DATA, cache_dir=E5_CACHE,
              item_texts=None, query_texts=None):
    """Для фиксированного benchmark сначала используются готовые top-5000."""
    paths = [cache_dir / f"top5000_{part}.npy" for part in ("indices", "scores")]
    if all(path.exists() for path in paths):
        try:
            arrays = tuple(np.load(path, mmap_mode="r") for path in paths)
            if all(array.shape == (len(queries), E5_TOP_K) for array in arrays):
                print("E5 top-5000: загружено")
                return arrays
        except (OSError, ValueError):
            pass  # Например, Git LFS скачал только файл-указатель.

    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    weights = E5_MODEL / "model.safetensors"
    if not weights.exists() or weights.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Нет локальных исходных весов E5: {E5_MODEL}. Выполните git lfs pull")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if item_texts is None or query_texts is None:
        raw_items = pd.read_parquet(data_dir / "benchmark_items.parquet",
                                    columns=["item_id", "item_title_raw", "item_description_raw"])
        raw_items = raw_items.assign(item_id=raw_items["item_id"].astype(str))
        raw_items = raw_items.set_index("item_id").reindex(items["item_id"].astype(str).to_numpy())
        raw_queries = pd.read_parquet(data_dir / "benchmark_queries.parquet",
                                      columns=["query_id", "search_query"])
        assert np.array_equal(raw_queries["query_id"].astype(str).to_numpy(),
                              queries["query_id"].astype(str).to_numpy())
        item_texts = ("passage: " + raw_items["item_title_raw"].fillna("").astype(str).str.strip()
                      + ". " + raw_items["item_description_raw"].fillna("").astype(str).str.strip()).to_numpy(str)
        query_texts = ("query: " + raw_queries["search_query"].fillna("").astype(str).str.strip()).to_numpy(str)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)
    model = SentenceTransformer(str(E5_MODEL), device=device)
    if device == "cuda":
        model.half()
    item_vectors = _embeddings(item_texts, cache_dir / "item_embeddings_maxlen128.npy", model, 8)
    query_vectors = _embeddings(query_texts, cache_dir / "query_embeddings_maxlen128.npy", model, 64)
    del model
    gc.collect()
    index_path = cache_dir / "hnsw_cosine.index"
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        assert index.ntotal == len(items)
    else:
        faiss.omp_set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        index = faiss.IndexHNSWFlat(item_vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        for first in range(0, len(items), 5000):
            index.add(np.ascontiguousarray(item_vectors[first:first + 5000], dtype=np.float32))
        faiss.write_index(index, str(index_path))
    index.hnsw.efSearch = 10000
    result = (np.empty((len(queries), E5_TOP_K), np.int32),
              np.empty((len(queries), E5_TOP_K), np.float32))
    for first in range(0, len(queries), 128):
        scores, indices = index.search(np.ascontiguousarray(query_vectors[first:first + 128],
                                                                 dtype=np.float32), E5_TOP_K)
        result[0][first:first + 128], result[1][first:first + 128] = indices, scores
    for path, array in zip(paths, result):
        np.save(path, array)
    return result
