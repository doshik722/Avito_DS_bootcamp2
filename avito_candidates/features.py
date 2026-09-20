"""Ровно 44 признака для финальных CatBoost и LightGBM."""

import numpy as np
import pandas as pd

from .config import CATBOOST_TOP_K, SOURCE_NAMES

NUMERIC = [
    "mixed_score", "hybrid_rank", "bm25_title_description", "char_tfidf_title",
    "bm25_title", "word_tfidf_title", "e5_cosine", "local_bm25", "geo_score",
    "same_location", "query_latitude", "query_longitude", "item_latitude",
    "item_longitude", "distance_km", "filter_match", "parameter_bm25",
    "log_price", "price_missing", "rating", "rating_missing", "log_reviews",
    "reviews_missing", "phone_hidden", "message_forbidden", "delivery",
    "has_query_coordinates",
]
CATEGORICAL = ["search_category", "item_category_id", "item_microcat_id"]
EXTRA = [
    "bm25_td_rank", "char_title_rank", "bm25_title_rank", "word_title_rank",
    "e5_rank", "parameter_rank", "filter_coverage", "has_filter",
    "query_in_title", "query_title_overlap", "query_word_count",
    "title_word_count", "description_chars", "price_vs_query_median",
]


def lookup_scores(candidates, indices, scores):
    """Нормированный скор источника для выбранных кандидатов."""
    valid = indices >= 0
    ids = np.asarray(indices[valid], dtype=np.int32)
    values = np.asarray(scores[valid], dtype=np.float32)
    result = np.zeros(len(candidates), dtype=np.float32)
    if not len(ids) or values[0] <= 0:
        return result
    order = np.argsort(ids)
    positions = np.searchsorted(ids[order], candidates)
    in_range = positions < len(ids)
    found = np.zeros(len(candidates), dtype=bool)
    found[in_range] = ids[order[positions[in_range]]] == candidates[in_range]
    result[found] = values[order[positions[found]]] / values[0]
    return result


def source_rank(candidates, source_indices):
    """Относительное место в источнике; 1.1 — объявление отсутствует."""
    source = np.asarray(source_indices)
    source = source[source >= 0]
    result = np.full(len(candidates), 1.1, dtype=np.float32)
    if not len(source):
        return result
    order = np.argsort(source)
    positions = np.searchsorted(source[order], candidates)
    in_range = positions < len(source)
    found = np.zeros(len(candidates), dtype=bool)
    found[in_range] = source[order[positions[in_range]]] == candidates[in_range]
    result[found] = (order[positions[found]] + 1) / len(source)
    return result


def build_features(data, geo, filters, candidates, mixed_scores, local_scores,
                   source_indices, source_scores, e5_indices, e5_scores,
                   parameter_indices, parameter_scores, top_k=CATBOOST_TOP_K):
    """Признаки считаются только для выбранных top-k, а не всего корпуса."""
    queries, items = data.queries, data.items
    count = len(queries)
    k = top_k
    numeric = np.empty((count * k, len(NUMERIC)), dtype=np.float32)
    categorical = np.empty((count * k, len(CATEGORICAL)), dtype=np.int32)
    extra = np.empty((count * k, len(EXTRA)), dtype=np.float32)

    item_lat = items["item_latitude"].astype("float64").to_numpy(np.float32)
    item_lon = items["item_longitude"].astype("float64").to_numpy(np.float32)
    query_centres = geo.centres.reindex(geo.query_locations).to_numpy(dtype=np.float32)
    price = pd.to_numeric(items["item_price"], errors="coerce").astype("float64")
    rating = pd.to_numeric(items["item_rating"], errors="coerce").astype("float64")
    reviews = pd.to_numeric(items["item_rating_reviews_count"], errors="coerce").astype("float64")
    log_price = np.log1p(price.fillna(0).clip(lower=0)).to_numpy(np.float32)
    log_reviews = np.log1p(reviews.fillna(0)).to_numpy(np.float32)
    price_missing = price.isna().to_numpy(np.float32)
    rating_missing = rating.isna().to_numpy(np.float32)
    reviews_missing = reviews.isna().to_numpy(np.float32)
    item_rating = rating.fillna(0).to_numpy(np.float32)
    phone = items["item_is_phone_hidden"].fillna(False).to_numpy(np.float32)
    message = items["item_is_message_forbidden"].fillna(False).to_numpy(np.float32)
    item_category = pd.to_numeric(items["item_category_id"], errors="coerce").fillna(-1).to_numpy(np.int32)
    microcat = pd.to_numeric(items["item_microcat_id"], errors="coerce").fillna(-1).to_numpy(np.int32)
    query_category = pd.to_numeric(queries["search_category"], errors="coerce").fillna(-1).to_numpy(np.int32)
    delivery = pd.to_numeric(queries["search_is_delivery_search"], errors="coerce").fillna(0).to_numpy(np.float32)
    titles = items["item_title_raw"].fillna("").astype(str).to_numpy()
    description_chars = items["item_description_raw"].fillna("").str.len().to_numpy(np.float32)
    query_texts = queries["search_query"].to_numpy()

    for row in range(count):
        chosen = candidates[row]
        start, stop = row * k, (row + 1) * k
        query_lat, query_lon = query_centres[row]
        lat, lon = item_lat[chosen], item_lon[chosen]
        lat_delta = np.deg2rad(lat - query_lat)
        lon_delta = np.deg2rad(lon - query_lon)
        distance_part = (np.sin(lat_delta / 2) ** 2
                         + np.cos(np.deg2rad(query_lat)) * np.cos(np.deg2rad(lat))
                         * np.sin(lon_delta / 2) ** 2)
        distance = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(distance_part, 0, 1)))
        filter_code = filters.codes[row]
        filter_match = (np.isin(chosen, filters.matches[filter_code]).astype(np.float32)
                        if filter_code >= 0 else np.zeros(k, np.float32))
        parameter_score = (lookup_scores(chosen, parameter_indices[filter_code],
                                         parameter_scores[filter_code])
                           if filter_code >= 0 else np.zeros(k, np.float32))
        text_scores = [lookup_scores(chosen, source_indices[name][row], source_scores[name][row])
                       for name in SOURCE_NAMES]
        numeric[start:stop] = np.column_stack([
            mixed_scores[row], np.arange(1, k + 1), *text_scores,
            lookup_scores(chosen, e5_indices[row], e5_scores[row]),
            local_scores[row], geo.score(row, chosen),
            (geo.item_locations[chosen] == geo.query_locations[row]).astype(np.float32),
            np.full(k, query_lat), np.full(k, query_lon), lat, lon, distance,
            filter_match, parameter_score, log_price[chosen], price_missing[chosen],
            item_rating[chosen], rating_missing[chosen], log_reviews[chosen],
            reviews_missing[chosen], phone[chosen], message[chosen],
            np.full(k, delivery[row]), np.full(k, np.isfinite(query_lat)),
        ])
        categorical[start:stop] = np.column_stack([
            np.full(k, query_category[row]), item_category[chosen], microcat[chosen],
        ])

        query = str(query_texts[row])
        query_words = set(query.split())
        chosen_titles = titles[chosen]
        title_words = [title.split() for title in chosen_titles]
        known_price = price_missing[chosen] == 0
        prices = log_price[chosen]
        median_price = np.median(prices[known_price]) if known_price.any() else 0
        parameter_rank = (source_rank(chosen, parameter_indices[filter_code])
                          if filter_code >= 0 else np.full(k, 1.1, np.float32))
        extra[start:stop] = np.column_stack([
            *[source_rank(chosen, source_indices[name][row]) for name in SOURCE_NAMES],
            source_rank(chosen, e5_indices[row]), parameter_rank,
            filters.coverage(row, chosen), np.full(k, filter_code >= 0),
            np.fromiter((query in title for title in chosen_titles), np.float32, k),
            np.fromiter((len(query_words.intersection(words)) / max(1, len(query_words))
                         for words in title_words), np.float32, k),
            np.full(k, len(query_words)),
            np.fromiter((len(words) for words in title_words), np.float32, k),
            description_chars[chosen], np.where(known_price, prices - median_price, 0),
        ])
        if (row + 1) % 500 == 0 or row + 1 == count:
            print(f"Признаки CatBoost: {row + 1:,}/{count:,}")

    frame = pd.DataFrame(numeric, columns=NUMERIC)
    for column, values in zip(CATEGORICAL, categorical.T):
        frame[column] = values
    for column, values in zip(EXTRA, extra.T):
        frame[column] = values
    return frame
