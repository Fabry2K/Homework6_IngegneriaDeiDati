import pandas as pd
from collections import defaultdict

# ==============================
# Impostazioni per stampa completa
# ==============================
pd.set_option("display.max_rows", None)        # stampa tutte le righe
pd.set_option("display.max_columns", None)     # stampa tutte le colonne
pd.set_option("display.width", None)           # nessun limite di larghezza
pd.set_option("display.max_colwidth", None)    # stampa completa del contenuto delle celle


# ==============================
# Schema mediato aggiornato
# ==============================
schema_mediato_columns = [
    "id",
    "vin",
    "manufacturer",
    "model",
    "year",
    "price",
    "mileage",
    "fuel_type",
    "transmission",
    "body_type",
    "cylinders",
    "invalid"
]

# ==============================
# Dizionari di rinomina interni
# ==============================
vehicles_rename = {
    "id": "id",
    "VIN": "vin",
    "manufacturer": "manufacturer",
    "model": "model",
    "year": "year",
    "price": "price",
    "odometer": "mileage",
    "fuel": "fuel_type",
    "transmission": "transmission",
    "type": "body_type",
    "cylinders":"cylinders"
}

used_cars_rename = {
    "listing_id": "id",
    "vin": "vin",
    "make_name": "manufacturer",
    "model_name": "model",
    "year": "year",
    "price": "price",
    "mileage": "mileage",
    "fuel_type": "fuel_type",
    "transmission": "transmission",
    "body_type": "body_type",
    "engine_cylinders":"cylinders"
}

# ==============================
# Funzione di allineamento a chunk
# ==============================
def align_dataset_in_chunks(input_path, output_path, dataset_type="vehicles", chunksize=200_000):
    """
    Legge il CSV in chunk, rinomina le colonne secondo lo schema mediato interno,
    seleziona solo le colonne dello schema mediato e salva in un CSV finale.
    
    dataset_type: "vehicles" o "used_cars"
    """
    if dataset_type == "vehicles":
        rename_dict = vehicles_rename
    elif dataset_type == "used_cars":
        rename_dict = used_cars_rename
    else:
        raise ValueError("dataset_type deve essere 'vehicles' o 'used_cars'")

    first_chunk = True

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        # rinomina colonne
        chunk_aligned = chunk.rename(columns=rename_dict)

        # seleziona solo le colonne presenti dello schema mediato
        existing_columns = [col for col in schema_mediato_columns if col in chunk_aligned.columns]
        chunk_aligned = chunk_aligned[existing_columns]

        # scrive su CSV
        chunk_aligned.to_csv(output_path, index=False, mode='w' if first_chunk else 'a', header=first_chunk)
        first_chunk = False

        print(f"Processed {len(chunk_aligned)} rows from {input_path}")

    print(f"Allineamento completato: {output_path}")


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