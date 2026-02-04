import pandas as pd
import math

# ===============================
# FUNZIONE PRINCIPALE
# ===============================
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

