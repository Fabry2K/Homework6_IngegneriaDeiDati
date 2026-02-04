import pandas as pd
import random

# ===============================
# CONFIGURAZIONE FILE E CHUNKSIZE
# ===============================
VEHICLES_ALIGNED_CSV = "vehicles_aligned.csv"
USED_CARS_ALIGNED_CSV = "used_cars_aligned.csv"
OUTPUT_CSV = "ground_truth.csv"
CHUNKSIZE = 100_000
NON_MATCHES_PER_MATCH = 5

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
    """
    Crea la ground-truth con match e non-match.
    Usa SOLO record con:
      - invalid == 0
      - vin presente
    """

    print("Creazione ground-truth in corso...")

    # ===============================
    # 1️⃣ Costruzione indice VIN -> used_cars_id
    # ===============================
    vin_to_used_ids = {}
    all_used_rows = []  # (id, vin) solo validi

    for chunk in pd.read_csv(used_cars_csv, chunksize=chunksize):
        # teniamo solo record validi
        chunk = chunk[
            (chunk["invalid"] == 0) &
            (chunk["vin"].notna()) &
            (chunk["vin"].astype(str).str.strip() != "")
        ]

        for _, row in chunk.iterrows():
            vin = row["vin"]
            u_id = row["id"]

            vin_to_used_ids.setdefault(vin, []).append(u_id)
            all_used_rows.append((u_id, vin))

    print(f"VIN distinti validi in used_cars: {len(vin_to_used_ids)}")
    print(f"Record used_cars validi: {len(all_used_rows)}")

    # ===============================
    # 2️⃣ Scansione vehicles e generazione match / non-match
    # ===============================
    first_chunk = True

    for chunk in pd.read_csv(vehicles_csv, chunksize=chunksize):
        rows = []

        # teniamo solo record validi
        chunk = chunk[
            (chunk["invalid"] == 0) &
            (chunk["vin"].notna()) &
            (chunk["vin"].astype(str).str.strip() != "")
        ]

        for _, v_row in chunk.iterrows():
            vin_v = v_row["vin"]
            v_id = v_row["id"]

            # -------- MATCH --------
            if vin_v in vin_to_used_ids:
                for u_id in vin_to_used_ids[vin_v]:
                    rows.append({
                        "vehicles_id": v_id,
                        "used_cars_id": u_id,
                        "label": 1
                    })

                # -------- NON-MATCH --------
                # generiamo non-match SOLO se esiste almeno un match
                candidates = [
                    u for u, vin_u in all_used_rows if vin_u != vin_v
                ]

                if len(candidates) >= non_matches_per_match:
                    sampled = random.sample(candidates, non_matches_per_match)
                    for u_id in sampled:
                        rows.append({
                            "vehicles_id": v_id,
                            "used_cars_id": u_id,
                            "label": 0
                        })

        # ===============================
        # Scrittura su CSV
        # ===============================
        if rows:
            df_out = pd.DataFrame(rows)
            df_out.to_csv(
                output_csv,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )
            first_chunk = False

    print(f"Ground-truth creata in '{output_csv}'")
    print(f"Rapporto non-match per match: {non_matches_per_match}")

