import pandas as pd
import random
from collections import Counter

# ===============================
# CONFIGURAZIONE
# ===============================
VEHICLES_ALIGNED_CSV = "vehicles_marked.csv"
USED_CARS_ALIGNED_CSV = "used_cars_marked.csv"

GROUND_TRUTH_CSV = "ground_truth.csv"
REVIEW_CSV = "ground_truth_review.csv"

CHUNKSIZE = 100_000
NON_MATCHES_PER_MATCH = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ===============================
# FUNZIONE PRINCIPALE
# ===============================
def create_ground_truth():
    print("Creazione ground-truth in corso...")

    # ===============================
    # 1️⃣ PRIMA PASSATA: VIN + ID
    # ===============================
    print("Analisi VIN e duplicati...")

    vehicles = []
    used_cars = []

    vehicles_vins = []
    used_vins = []

    for chunk in pd.read_csv(VEHICLES_ALIGNED_CSV, chunksize=CHUNKSIZE):
        chunk = chunk[
            (chunk["invalid"] == 0) &
            (chunk["vin"].notna()) &
            (chunk["vin"].astype(str).str.strip() != "")
        ]
        for _, r in chunk.iterrows():
            vehicles.append((r["id"], r["vin"]))
            vehicles_vins.append(r["vin"])

    for chunk in pd.read_csv(USED_CARS_ALIGNED_CSV, chunksize=CHUNKSIZE):
        chunk = chunk[
            (chunk["invalid"] == 0) &
            (chunk["vin"].notna()) &
            (chunk["vin"].astype(str).str.strip() != "")
        ]
        for _, r in chunk.iterrows():
            used_cars.append((r["id"], r["vin"]))
            used_vins.append(r["vin"])

    dup_vehicles_vins = {vin for vin, c in Counter(vehicles_vins).items() if c > 1}
    dup_used_vins = {vin for vin, c in Counter(used_vins).items() if c > 1}

    print(f"Vehicles validi: {len(vehicles)}")
    print(f"Used cars validi: {len(used_cars)}")
    print(f"VIN duplicati vehicles: {len(dup_vehicles_vins)}")
    print(f"VIN duplicati used_cars: {len(dup_used_vins)}")

    # ===============================
    # 2️⃣ INDICE used_cars per VIN
    # ===============================
    vin_to_used_ids = {}
    for u_id, vin in used_cars:
        vin_to_used_ids.setdefault(vin, []).append(u_id)

    # ===============================
    # 3️⃣ FILE OUTPUT
    # ===============================
    gt_written = False
    review_written = False

    gt_count = 0
    review_count = 0
    non_match_count = 0

    # ===============================
    # 4️⃣ MATCH
    # ===============================
    print("Generazione match...")

    for v_id, vin_v in vehicles:
        if vin_v not in vin_to_used_ids:
            continue

        for u_id in vin_to_used_ids[vin_v]:
            is_dup = (vin_v in dup_vehicles_vins) or (vin_v in dup_used_vins)

            # ===============================
            # MATCH DA REVISIONARE
            # ===============================
            if is_dup:
                v_rec = pd.read_csv(
                    VEHICLES_ALIGNED_CSV,
                    chunksize=CHUNKSIZE
                ).apply(lambda df: df[df["id"] == v_id]).dropna(how="all")

                u_rec = pd.read_csv(
                    USED_CARS_ALIGNED_CSV,
                    chunksize=CHUNKSIZE
                ).apply(lambda df: df[df["id"] == u_id]).dropna(how="all")

                if not v_rec.empty and not u_rec.empty:
                    out = pd.concat(
                        [v_rec.reset_index(drop=True), u_rec.reset_index(drop=True)],
                        axis=1,
                        keys=["vehicles", "used_cars"]
                    )
                    out.to_csv(
                        REVIEW_CSV,
                        mode="a",
                        header=not review_written,
                        index=False
                    )
                    review_written = True
                    review_count += 1
                continue

            # ===============================
            # MATCH VALIDO
            # ===============================
            pd.DataFrame(
                [{
                    "vehicles_id": v_id,
                    "used_cars_id": u_id,
                    "label": 1
                }]
            ).to_csv(
                GROUND_TRUTH_CSV,
                mode="a",
                header=not gt_written,
                index=False
            )
            gt_written = True
            gt_count += 1

            # ===============================
            # NON-MATCH CASUALI
            # ===============================
            generated = 0
            while generated < NON_MATCHES_PER_MATCH:
                v_nm, vin_nm_v = random.choice(vehicles)
                u_nm, vin_nm_u = random.choice(used_cars)

                if vin_nm_v == vin_nm_u:
                    continue
                if v_nm == v_id and u_nm == u_id:
                    continue

                pd.DataFrame(
                    [{
                        "vehicles_id": v_nm,
                        "used_cars_id": u_nm,
                        "label": 0
                    }]
                ).to_csv(
                    GROUND_TRUTH_CSV,
                    mode="a",
                    header=False,
                    index=False
                )
                generated += 1
                non_match_count += 1

    # ===============================
    # 5️⃣ RIEPILOGO
    # ===============================
    print("✅ Ground-truth completata")
    print(f"Match validi: {gt_count}")
    print(f"Non-match: {non_match_count}")
    print(f"Match da revisionare: {review_count}")

    if review_count == 0:
        print("ℹ️ Nessun match con VIN duplicati: file review non creato")


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    create_ground_truth()
