"""География и фильтры: те же сигналы, что в финальном ноутбуке."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from .config import (
    CITY_GEO_WEIGHT, FILTER_BONUS, FILTER_THRESHOLD,
    LOCAL_RADIUS_KM, REGION_GEO_WEIGHT,
)


@dataclass
class Geography:
    centres: pd.DataFrame
    city_ids: pd.Index
    distances: np.ndarray
    item_city_codes: np.ndarray
    query_region_codes: np.ndarray
    region_scores: np.ndarray
    query_locations: np.ndarray
    item_locations: np.ndarray

    @classmethod
    def from_data(cls, data):
        locations = pd.concat([
            data.train_items[["item_location_id", "item_latitude", "item_longitude"]],
            data.items[["item_location_id", "item_latitude", "item_longitude"]],
        ], ignore_index=True).astype({"item_latitude": "float64", "item_longitude": "float64"})
        centres = (locations.dropna(subset=["item_location_id", "item_latitude", "item_longitude"])
                   .groupby("item_location_id")[["item_latitude", "item_longitude"]]
                   .median().sort_index())
        city_ids = pd.Index(centres.index)
        lat = np.deg2rad(centres["item_latitude"].to_numpy())
        lon = np.deg2rad(centres["item_longitude"].to_numpy())
        vectors = np.column_stack([np.cos(lat) * np.cos(lon),
                                   np.cos(lat) * np.sin(lon), np.sin(lat)]).astype(np.float32)
        distances = (np.arccos(np.clip(vectors @ vectors.T, -1, 1)) * 6371.0088).astype(np.float32)
        np.fill_diagonal(distances, 0)
        item_locations = data.items["item_location_id"].to_numpy()
        query_locations = data.queries["search_location_id"].to_numpy()
        item_codes = city_ids.get_indexer(item_locations)

        # Локации без координатного центра сопоставляем городам по кликам train.
        pairs = (data.relevance
                 .merge(data.train_queries[["train_query_id", "search_location_id"]],
                        on="train_query_id", validate="many_to_one")
                 .merge(data.train_items[["item_id", "item_location_id"]],
                        on="item_id", validate="many_to_one"))
        query_codes = city_ids.get_indexer(pairs["search_location_id"])
        clicked_city_codes = city_ids.get_indexer(pairs["item_location_id"])
        region_mask = (query_codes < 0) & (clicked_city_codes >= 0)
        region_ids = pd.Index(pd.unique(pairs.loc[region_mask, "search_location_id"].dropna()))
        region_codes = region_ids.get_indexer(pairs.loc[region_mask, "search_location_id"])
        counts = np.zeros((len(region_ids), len(city_ids)), dtype=np.int32)
        np.add.at(counts, (region_codes, clicked_city_codes[region_mask]), 1)
        region_scores = np.zeros_like(counts, dtype=np.float32)
        for code, city_counts in enumerate(counts):
            maximum = city_counts.max()
            if maximum:
                region_scores[code] = np.log1p(city_counts) / np.log1p(maximum)
        return cls(centres, city_ids, distances, item_codes,
                   region_ids.get_indexer(query_locations), region_scores,
                   query_locations, item_locations)

    def score(self, row, candidates):
        codes = self.item_city_codes[candidates]
        query_code = self.city_ids.get_indexer([self.query_locations[row]])[0]
        result = np.zeros(len(candidates), dtype=np.float32)
        known = codes >= 0
        if query_code >= 0:
            bins = np.full(len(candidates), 3, dtype=np.int8)
            distances = self.distances[query_code, codes[known]]
            bins[known] = np.select([
                codes[known] == query_code, distances <= 25, distances <= LOCAL_RADIUS_KM,
            ], [0, 1, 2], default=3).astype(np.int8)
            return (CITY_GEO_WEIGHT * np.array([1.0, 0.55, 0.25, 0.0])[bins]).astype(np.float32)
        region_code = self.query_region_codes[row]
        if 0 <= region_code < len(self.region_scores):
            result[known] = REGION_GEO_WEIGHT * self.region_scores[region_code, codes[known]]
        return result

    def groups(self):
        """Для каждого города — радиус 75 км, для региона — связанные города."""
        for location in pd.unique(self.query_locations):
            if pd.isna(location):
                continue
            rows = np.flatnonzero(self.query_locations == location)
            city = self.city_ids.get_indexer([location])[0]
            if city >= 0:
                nearby = np.flatnonzero(self.distances[city] <= LOCAL_RADIUS_KM)
                item_rows = np.flatnonzero(np.isin(self.item_city_codes, nearby))
            else:
                region = self.query_region_codes[rows[0]]
                known = self.item_city_codes >= 0
                mask = np.zeros(len(self.item_city_codes), dtype=bool)
                if region >= 0:
                    mask[known] = self.region_scores[region, self.item_city_codes[known]] > 0
                item_rows = np.flatnonzero(mask)
            if len(item_rows):
                yield rows, item_rows


@dataclass
class Filters:
    texts: pd.Index
    codes: np.ndarray
    item_tokens: object
    query_tokens: object
    counts: np.ndarray
    matches: dict

    @classmethod
    def from_data(cls, queries, items):
        values = queries["search_infm_params_text"].fillna("")
        texts = pd.Index(values.loc[values.ne("")].unique())
        codes = texts.get_indexer(values)
        vectorizer = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")
        vectorizer.fit(pd.concat([items["item_infm_params_text"].fillna(""), pd.Series(texts)]))
        item_tokens = vectorizer.transform(items["item_infm_params_text"].fillna(""))
        query_tokens = vectorizer.transform(texts)
        counts = np.asarray(query_tokens.sum(axis=1)).ravel()
        required = np.ceil(FILTER_THRESHOLD * counts).astype(np.int32)
        matches = {}
        for code in range(len(texts)):
            if required[code] == 0:
                matches[code] = np.arange(len(items), dtype=np.int32)
            else:
                overlap = (query_tokens[code] @ item_tokens.T).tocsr()
                matches[code] = overlap.indices[overlap.data >= required[code]]
        return cls(texts, codes, item_tokens, query_tokens, counts, matches)

    def bonus(self, row, candidates):
        code = self.codes[row]
        if code < 0:
            return np.zeros(len(candidates), dtype=np.float32)
        return FILTER_BONUS * np.isin(candidates, self.matches[code], assume_unique=True)

    def coverage(self, row, candidates):
        code = self.codes[row]
        if code < 0:
            return np.zeros(len(candidates), dtype=np.float32)
        return ((self.query_tokens[code] @ self.item_tokens[candidates].T)
                .toarray().ravel() / max(1, self.counts[code])).astype(np.float32)
