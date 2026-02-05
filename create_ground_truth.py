import pandas as pd
import random

# ===============================
# CONFIGURAZIONE
# ===============================
VEHICLES_CSV = "vehicles_marked.csv"
USED_CARS_CSV = "used_cars_marked.csv"
OUTPUT_CSV = "ground_truth_full.csv"

CHUNKSIZE = 100_000
NON_MATCHES_PER_MATCH = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ===============================
# FUNZIONE PRINCIPALE
# ===============================
def create_ground_truth(
    vehicles_csv=VEHICLES_CSV,
    used_cars_csv=USED_CARS_CSV,
    output_csv=OUTPUT_CSV,
    chunksize=CHUNKSIZE,
    non_matches_per_match=NON_MATCHES_PER_MATCH
):
    print("Creazione ground-truth (record completi) in corso...")

    # ===============================
    # 1️⃣ CARICAMENTO USED_CARS VALIDI
    # ===============================
    print("Indicizzazione used_cars validi...")

    used_rows = {}   # id -> record completo
    vin_to_used_ids = {}

    for chunk in pd.read_csv(used_cars_csv, chunksize=chunksize):
        chunk = chunk[
            (chunk["invalid"] == 0) &
            (chunk["vin"].notna()) &
            (chunk["vin"].astype(str).str.strip() != "")
        ]

        for _, row in chunk.iterrows():
            u_id = row["id"]
            vin = row["vin"]
            record = row.to_dict()

            used_rows[u_id] = record
            vin_to_used_ids.setdefault(vin, []).append(u_id)

    print(f"Used cars validi indicizzati: {len(used_rows)}")

    if not used_rows:
        print("⚠️ Nessun used_cars valido trovato.")
        return

    used_ids = list(used_rows.keys())

    # ===============================
    # 2️⃣ SCANSIONE VEHICLES + MATCH
    # ===============================
    print("Scansione vehicles e generazione coppie...")

    first_write = True
    total_matches = 0
    total_non_matches = 0

    for chunk in pd.read_csv(vehicles_csv, chunksize=chunksize):
        chunk = chunk[
            (chunk["invalid"] == 0) &
            (chunk["vin"].notna()) &
            (chunk["vin"].astype(str).str.strip() != "")
        ]

        output_rows = []

        for _, v_row in chunk.iterrows():
            v_id = v_row["id"]
            vin_v = v_row["vin"]
            v_record = v_row.to_dict()

            # ================= MATCH =================
            if vin_v in vin_to_used_ids:
                for u_id in vin_to_used_ids[vin_v]:
                    u_record = used_rows[u_id]

                    row = {}

                    # prefix vehicles_
                    for k, v in v_record.items():
                        row[f"vehicles_{k}"] = v

                    # prefix used_cars_
                    for k, v in u_record.items():
                        row[f"used_cars_{k}"] = v

                    row["label"] = 1
                    output_rows.append(row)
                    total_matches += 1

                    # ============ NON-MATCH CASUALI ============
                    non_match_count = 0
                    while non_match_count < non_matches_per_match:
                        rand_u_id = random.choice(used_ids)
                        rand_u_record = used_rows[rand_u_id]

                        if rand_u_record["vin"] == vin_v:
                            continue

                        nm_row = {}

                        for k, v in v_record.items():
                            nm_row[f"vehicles_{k}"] = v
                        for k, v in rand_u_record.items():
                            nm_row[f"used_cars_{k}"] = v

                        nm_row["label"] = 0
                        output_rows.append(nm_row)

                        non_match_count += 1
                        total_non_matches += 1

        # ================= SCRITTURA CHUNK =================
        if output_rows:
            df_out = pd.DataFrame(output_rows)
            df_out.to_csv(
                output_csv,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )
            first_write = False

    print("✅ Ground-truth creata con successo")
    print(f"Match: {total_matches}")
    print(f"Non-match: {total_non_matches}")
    print(f"Totale coppie: {total_matches + total_non_matches}")


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    create_ground_truth()
