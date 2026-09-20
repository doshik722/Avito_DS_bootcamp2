from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
BENCHMARK_CACHE = ARTIFACTS / "retrieval" / "final_benchmark"
E5_CACHE = ARTIFACTS / "retrieval" / "benchmark_mixed_geo_e5"
E5_MODEL = ARTIFACTS / "models" / "multilingual-e5-base"
CATBOOST_MODEL = ARTIFACTS / "models" / "catboost_extra_all4500_top300.cbm"
LIGHTGBM_MODEL = ARTIFACTS / "models" / "lightgbm_all4500_top300.zip"

# Зафиксированы по лучшему отправленному ответу, Recall@50 = 0.841578.
SOURCE_NAMES = (
    "bm25_title_description", "char_tfidf_title",
    "bm25_title", "word_tfidf_title",
)
SOURCE_WEIGHTS = (0.55, 0.40, 0.05, 0.0)
TEXT_TOP_K = E5_TOP_K = 5000
LOCAL_TOP_K = 3000
CATBOOST_TOP_K = 300
ANSWER_TOP_K = 50
LOCAL_RADIUS_KM = 75.0
CITY_GEO_WEIGHT = 1.50
REGION_GEO_WEIGHT = 0.75
FILTER_THRESHOLD = 0.90
FILTER_BONUS = 0.55
COSINE_WEIGHT = LOCAL_WEIGHT = 0.05
CATBOOST_WEIGHT = 0.70
LIGHTGBM_WEIGHT = 0.40
LIGHTGBM_TREES = 200
