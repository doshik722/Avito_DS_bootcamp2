"""Создать answer.csv. Готовые артефакты загружаются автоматически."""

import argparse
from pathlib import Path

from avito_candidates.config import ROOT
from avito_candidates.pipeline import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "answer.csv")
    parser.add_argument("--verify", action="store_true",
                        help="сравнить выдачу с CSV, давшим Recall@50=0.841578")
    args = parser.parse_args()
    answer = predict()
    if args.verify:
        reference = ROOT / "answer_catboost_lightgbm_top300.csv"
        import pandas as pd
        old = pd.read_csv(reference, dtype=str)
        assert answer.equals(old), "Выдача отличается от проверенного файла"
        print("Совпадение с проверенным CSV: 100%")
    answer.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Создан: {args.output.resolve()}")


if __name__ == "__main__":
    main()
