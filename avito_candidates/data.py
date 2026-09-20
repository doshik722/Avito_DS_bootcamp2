from dataclasses import dataclass

import pandas as pd

from .config import DATA

TEXT_COLUMNS = (
    "search_query", "search_infm_params_text", "item_title_raw",
    "item_infm_params_text", "item_description_raw",
)
SERVICE_LABELS = (
    "вид услуги", "тип услуги", "место оказания услуг",
    "название услуги", "марка авто",
)


@dataclass
class Datasets:
    queries: pd.DataFrame
    items: pd.DataFrame
    train_queries: pd.DataFrame
    train_items: pd.DataFrame
    relevance: pd.DataFrame


def normalize_text(series):
    """Та же нормализация, что использовалась в финальном ноутбуке."""
    return (series.fillna("").str.lower()
            .str.replace("ё", "е", regex=False)
            .str.replace(r"[^0-9a-zа-я]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def prepare_text(frame):
    for column in TEXT_COLUMNS:
        if column in frame:
            frame[column] = normalize_text(frame[column])
    for column in ("search_infm_params_text", "item_infm_params_text"):
        if column in frame:
            text = frame[column]
            for label in SERVICE_LABELS:
                text = text.str.replace(label, "", regex=False)
            frame[column] = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return frame


def load_data(data_dir=DATA):
    """Train нужен для географии; benchmark — корпус поиска и запросы."""
    train = prepare_text(pd.read_parquet(data_dir / "train.parquet", dtype_backend="pyarrow"))
    queries = prepare_text(pd.read_parquet(data_dir / "benchmark_queries.parquet", dtype_backend="pyarrow"))
    items = prepare_text(pd.read_parquet(data_dir / "benchmark_items.parquet", dtype_backend="pyarrow"))
    query_columns = queries.columns.drop("query_id").tolist()
    train_queries = train[query_columns].drop_duplicates().reset_index(drop=True)
    train_queries.insert(0, "train_query_id", range(len(train_queries)))
    train_items = train[items.columns].drop_duplicates("item_id").reset_index(drop=True)
    relevance = (train[query_columns + ["item_id"]]
                 .merge(train_queries, on=query_columns, how="left")
                 [["train_query_id", "item_id"]].drop_duplicates().reset_index(drop=True))
    return Datasets(queries, items, train_queries, train_items, relevance)
