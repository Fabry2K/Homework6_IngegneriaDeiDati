import pandas as pd

# ==============================
# Schema mediato aggiornato
# ==============================
schema_mediato_columns = [
    "vin",
    "manufacturer",
    "model",
    "year",
    "mileage",
    "fuel_type",
    "transmission",
    "body_type",
    "cylinders",
    "drive",
    "color",
    "invalid"
]

# ==============================
# Dizionari di rinomina interni
# ==============================
vehicles_rename = {
    "VIN": "vin",
    "manufacturer": "manufacturer",
    "model": "model",
    "year": "year",
    "odometer": "mileage",
    "fuel": "fuel_type",
    "transmission": "transmission",
    "type": "body_type",
    "cylinders": "cylinders",
    "drive": "drive",
    "paint_color": "color",
}

used_cars_rename = {
    "vin": "vin",
    "make_name": "manufacturer",
    "model_name": "model",
    "year": "year",
    "mileage": "mileage",
    "fuel_type": "fuel_type",
    "transmission": "transmission",
    "body_type": "body_type",
    "engine_cylinders": "cylinders",
    "wheel_system": "drive",
    "exterior_color": "color"
}

# ==============================
# Funzione di allineamento a chunk
# ==============================
def align_dataset(input_path, output_path, dataset_type="vehicles", chunksize=50_000):
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

        # 👉 AGGIUNTA: crea la colonna invalid (sempre 0)
        chunk_aligned["invalid"] = 0

        # seleziona solo le colonne dello schema mediato
        chunk_aligned = chunk_aligned[schema_mediato_columns]

        # scrive su CSV
        chunk_aligned.to_csv(
            output_path,
            index=False,
            mode='w' if first_chunk else 'a',
            header=first_chunk
        )

        first_chunk = False
        print(f"Processed {len(chunk_aligned)} rows from {input_path}")

    print(f"Allineamento completato: {output_path}")
