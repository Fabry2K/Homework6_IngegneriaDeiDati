import pandas as pd
import random

# ===============================
# CONFIGURAZIONE
# ===============================
VEHICLES_ALIGNED_CSV = "vehicles_aligned.csv"
USED_CARS_ALIGNED_CSV = "used_cars_aligned.csv"
OUTPUT_CSV = "ground_truth.csv"

CHUNKSIZE = 100_000
NON_MATCHES_PER_MATCH = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ===============================
# FUNZIONE PRINCIPALE
# ===============================
def create_ground_truth(
    vehicles_csv=VEHICLES_ALIGNED_CSV,
    used_cars_csv=USED_CARS_ALIGNED_CSV,
    output_csv=OUTPUT_CSV,
    chunksize=CHUNKSIZE,
    non_matches_per_match=NON_MATCHES_PER_MATCH
):
    print("Creazione ground-truth in corso...")

    # ===============================
    # 1️⃣ CARICAMENTO RECORD VALIDi
    # ===============================
    print("Caricamento record validi...")

    vehicles_valid = []
    used_cars_valid = []

    # --- vehicles ---
    for chunk in pd.read_csv(vehicles_csv, chunksize=chunksize):
        chunk = chunk[
            (chunk["invalid"] == 0) &
            (chunk["vin"].notna()) &
            (chunk["vin"].astype(str).str.strip() != "")
        ]

        vehicles_valid.extend(
            [(row["id"], row["vin"]) for _, row in chunk.iterrows()]
        )

    # --- used_cars ---
    for chunk in pd.read_csv(used_cars_csv, chunksize=chunksize):
        chunk = chunk[
            (chunk["invalid"] == 0) &
            (chunk["vin"].notna()) &
            (chunk["vin"].astype(str).str.strip() != "")
        ]

        used_cars_valid.extend(
            [(row["id"], row["vin"]) for _, row in chunk.iterrows()]
        )

    print(f"Vehicles validi: {len(vehicles_valid)}")
    print(f"Used cars validi: {len(used_cars_valid)}")

    # ===============================
    # 2️⃣ GENERAZIONE MATCH
    # ===============================
    print("Generazione match...")

    vin_to_used_ids = {}
    for u_id, vin in used_cars_valid:
        vin_to_used_ids.setdefault(vin, []).append(u_id)

    matches = []
    for v_id, vin_v in vehicles_valid:
        if vin_v in vin_to_used_ids:
            for u_id in vin_to_used_ids[vin_v]:
                matches.append((v_id, u_id, 1))

    print(f"Match trovati: {len(matches)}")

    if len(matches) == 0:
        print("⚠️ Nessun match trovato. Ground-truth non creata.")
        return

    # ===============================
    # 3️⃣ GENERAZIONE NON-MATCH CASUALI
    # ===============================
    print("Generazione non-match casuali...")

    target_non_matches = len(matches) * non_matches_per_match
    non_matches = set()

    match_pairs = {(v, u) for v, u, _ in matches}

    while len(non_matches) < target_non_matches:
        v_id, vin_v = random.choice(vehicles_valid)
        u_id, vin_u = random.choice(used_cars_valid)

        # VIN diversi
        if vin_v == vin_u:
            continue

        # Evita collisioni con match veri
        if (v_id, u_id) in match_pairs:
            continue

        non_matches.add((v_id, u_id, 0))

        if len(non_matches) % 100_000 == 0:
            print(f"  Non-match generati: {len(non_matches)} / {target_non_matches}")

    print(f"Non-match generati: {len(non_matches)}")

    # ===============================
    # 4️⃣ SCRITTURA OUTPUT
    # ===============================
    print("Scrittura file di output...")

    df_out = pd.DataFrame(
        matches + list(non_matches),
        columns=["vehicles_id", "used_cars_id", "label"]
    )

    df_out.to_csv(output_csv, index=False)

    print(f"✅ Ground-truth creata in '{output_csv}'")
    print(f"Totale coppie: {len(df_out)}")
    print(f"Rapporto non-match/match: {non_matches_per_match}:1")


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    create_ground_truth()
