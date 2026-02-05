import pandas as pd
import math
import random

def check_representativity(
    input_csv,
    chunksize=100_000
):
    print(f"\nAnalisi rappresentatività dataset: {input_csv}\n")

    years = set()
    manufacturers = set()
    models = set()

    price_min, price_max, price_sum, price_count = math.inf, -math.inf, 0, 0
    mileage_min, mileage_max, mileage_sum, mileage_count = math.inf, -math.inf, 0, 0

    total_rows = 0
    vin_present = 0

    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        total_rows += len(chunk)

        # --- VIN ---
        if "vin" in chunk.columns:
            vin_present += chunk["vin"].notna().sum()

        # --- YEAR ---
        if "year" in chunk.columns:
            years.update(chunk["year"].dropna().unique())

        # --- MANUFACTURER ---
        if "manufacturer" in chunk.columns:
            manufacturers.update(chunk["manufacturer"].dropna().unique())

        # --- MODEL ---
        if "model" in chunk.columns:
            models.update(chunk["model"].dropna().unique())

        # --- PRICE ---
        if "price" in chunk.columns:
            prices = chunk["price"].dropna()
            if not prices.empty:
                price_min = min(price_min, prices.min())
                price_max = max(price_max, prices.max())
                price_sum += prices.sum()
                price_count += len(prices)

        # --- MILEAGE ---
        if "mileage" in chunk.columns:
            mileages = chunk["mileage"].dropna()
            if not mileages.empty:
                mileage_min = min(mileage_min, mileages.min())
                mileage_max = max(mileage_max, mileages.max())
                mileage_sum += mileages.sum()
                mileage_count += len(mileages)

    # ===============================
    # PRINT RISULTATI
    # ===============================
    print("=== Copertura attributi ===")
    print(f"Record totali: {total_rows}")
    print(f"VIN presenti: {vin_present} ({(vin_present / total_rows) * 100:.2f}%)")

    print("\n=== YEAR ===")
    print(f"Anni distinti: {len(years)}")
    print(f"Range anni: {min(years)} - {max(years)}" if years else "N/A")

    print("\n=== MANUFACTURER ===")
    print(f"Produttori distinti: {len(manufacturers)}")

    print("\n=== MODEL ===")
    print(f"Modelli distinti: {len(models)}")

    print("\n=== PRICE ===")
    if price_count > 0:
        print(f"Min: {price_min}")
        print(f"Max: {price_max}")
        print(f"Media: {price_sum / price_count:.2f}")
    else:
        print("Nessun dato disponibile")

    print("\n=== MILEAGE ===")
    if mileage_count > 0:
        print(f"Min: {mileage_min}")
        print(f"Max: {mileage_max}")
        print(f"Media: {mileage_sum / mileage_count:.2f}")
    else:
        print("Nessun dato disponibile")

    print("\nAnalisi completata.\n")

import pandas as pd

# ===============================
# CONFIGURAZIONE GENERALE
# ===============================
CHUNKSIZE = 100_000


# ==========================================================
# 1️⃣ RIMOZIONE VIN DAI DATASET ALIGNED
# ==========================================================
def remove_vin(
    vehicles_input,
    used_cars_input,
    vehicles_output,
    used_cars_output,
    chunksize=CHUNKSIZE
):
    """
    Rimuove la colonna VIN dai file aligned (vehicles e used_cars)
    lavorando a chunk per non saturare la RAM.
    """

    def _process(input_csv, output_csv):
        first_chunk = True

        for chunk in pd.read_csv(input_csv, chunksize=chunksize):
            if "vin" in chunk.columns:
                chunk = chunk.drop(columns=["vin"])

            chunk.to_csv(
                output_csv,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            first_chunk = False

        print(f"✅ Creato file senza VIN: {output_csv}")

    _process(vehicles_input, vehicles_output)
    _process(used_cars_input, used_cars_output)


# ==========================================================
# 2️⃣ RIMOZIONE VIN DALLA GROUND TRUTH
# ==========================================================
def remove_vin_from_ground_truth(
    ground_truth_input,
    ground_truth_output,
    chunksize=CHUNKSIZE
):
    """
    Rimuove vehicles_vin e used_cars_vin dalla ground truth.
    """

    first_chunk = True

    for chunk in pd.read_csv(ground_truth_input, chunksize=chunksize):

        cols_to_drop = []
        if "vehicles_vin" in chunk.columns:
            cols_to_drop.append("vehicles_vin")
        if "used_cars_vin" in chunk.columns:
            cols_to_drop.append("used_cars_vin")

        if cols_to_drop:
            chunk = chunk.drop(columns=cols_to_drop)

        chunk.to_csv(
            ground_truth_output,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

    print(f"✅ Ground truth senza VIN creata: {ground_truth_output}")



# ==========================================================
# 3️⃣ SPLIT GROUND TRUTH IN TRAIN / VAL / TEST
# ==========================================================
def split_ground_truth(
    ground_truth_input,
    train_output,
    val_output,
    test_output,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    chunksize=CHUNKSIZE
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    print("📂 Split ground truth in train / val / test...")

    # writer state
    writers = {
        "train": {"path": train_output, "first": True},
        "val": {"path": val_output, "first": True},
        "test": {"path": test_output, "first": True},
    }

    def _write(df, key):
        w = writers[key]
        df.to_csv(
            w["path"],
            mode="w" if w["first"] else "a",
            header=w["first"],
            index=False
        )
        w["first"] = False

    for chunk in pd.read_csv(ground_truth_input, chunksize=chunksize):

        # split stratificato sulla label
        for label_value, group in chunk.groupby("label"):

            indices = list(group.index)
            random.shuffle(indices)

            n = len(indices)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            if train_idx:
                _write(group.loc[train_idx], "train")
            if val_idx:
                _write(group.loc[val_idx], "val")
            if test_idx:
                _write(group.loc[test_idx], "test")

    print("✅ Split completato:")
    print(f"  • Train → {train_output}")
    print(f"  • Val   → {val_output}")
    print(f"  • Test  → {test_output}")