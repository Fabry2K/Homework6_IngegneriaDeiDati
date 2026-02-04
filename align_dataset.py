import pandas as pd

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
