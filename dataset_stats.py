import pandas as pd
from collections import defaultdict

def count_nulls_and_uniques_big_csv(path, name, chunksize=50_000):
    null_counts = None
    unique_sets = defaultdict(set)
    total_rows = 0

    for chunk in pd.read_csv(path, chunksize=chunksize):
        if null_counts is None:
            null_counts = chunk.isnull().sum()
        else:
            null_counts += chunk.isnull().sum()

        for col in chunk.columns:
            unique_sets[col].update(chunk[col].dropna().unique())

        total_rows += len(chunk)

    report = pd.DataFrame({
        "null_count": null_counts,
        "null_%": (null_counts / total_rows) * 100,
        "unique_values": {col: len(vals) for col, vals in unique_sets.items()}
    })

    report["null_%"] = report["null_%"].round(2)
    report = report.sort_values(by="null_%", ascending=False)

    print("=" * 80)
    print(f"DATASET: {name}")
    print(f"Numero di record: {total_rows}")
    print(f"Numero di attributi: {len(report)}")
    print("=" * 80)
    print(report)

    return report