import pandas as pd
import random

def build_ground_truth(
    file_a,
    file_b,
    output_gt,
    chunksize=200_000,
    negatives_per_match=2,
    random_seed=42
):
    random.seed(random_seed)

    print("📥 Caricamento file A (in RAM)...")
    df_a = pd.read_csv(file_a)
    df_a = df_a[df_a["invalid"] == 0].reset_index(drop=True)
    print(f"✔ Record validi in A: {len(df_a)}")

    # colonne da scrivere (escludo invalid)
    cols_a = [c for c in df_a.columns if c != "invalid"]

    first_write = True
    match_count = 0
    nonmatch_count = 0

    print("🚀 Inizio scansione file B a chunk...")

    for chunk_idx, df_b in enumerate(pd.read_csv(file_b, chunksize=chunksize)):
        df_b = df_b[df_b["invalid"] == 0].reset_index(drop=True)
        if df_b.empty:
            continue

        cols_b = [c for c in df_b.columns if c != "invalid"]

        print(f"➡ Chunk {chunk_idx} | record validi in B: {len(df_b)}")

        # indicizzazione veloce su VIN
        b_by_vin = {vin: g for vin, g in df_b.groupby("vin")}

        rows_out = []

        for _, row_a in df_a.iterrows():
            vin = row_a["vin"]

            if vin not in b_by_vin:
                continue

            # prendo un match qualsiasi
            row_b = b_by_vin[vin].iloc[0]

            # MATCH
            match_row = {}
            for c in cols_a:
                match_row[f"a_{c}"] = row_a[c]
            for c in cols_b:
                match_row[f"b_{c}"] = row_b[c]
            match_row["match"] = 1

            rows_out.append(match_row)
            match_count += 1

            # ===== NON MATCH =====
            half = negatives_per_match // 2
            rest = negatives_per_match - half

            # 1️⃣ HARD NON MATCH: stesso VIN A, VIN diverso in B
            b_diff = df_b[df_b["vin"] != vin]
            for _ in range(min(half, len(b_diff))):
                row_b_nm = b_diff.sample(1).iloc[0]

                nm = {}
                for c in cols_a:
                    nm[f"a_{c}"] = row_a[c]
                for c in cols_b:
                    nm[f"b_{c}"] = row_b_nm[c]
                nm["match"] = 0

                rows_out.append(nm)
                nonmatch_count += 1

            # 2️⃣ RANDOM NON MATCH: VIN diverso in A, stesso VIN in B
            a_diff = df_a[df_a["vin"] != vin]
            for _ in range(min(rest, len(a_diff))):
                row_a_nm = a_diff.sample(1).iloc[0]

                nm = {}
                for c in cols_a:
                    nm[f"a_{c}"] = row_a_nm[c]
                for c in cols_b:
                    nm[f"b_{c}"] = row_b[c]
                nm["match"] = 0

                rows_out.append(nm)
                nonmatch_count += 1

        if rows_out:
            pd.DataFrame(rows_out).to_csv(
                output_gt,
                mode="w" if first_write else "a",
                index=False,
                header=first_write
            )
            first_write = False

    print("\n✅ Ground truth completata")
    print(f"✔ Match generati: {match_count}")
    print(f"✔ Non-match generati: {nonmatch_count}")
    print(f"📁 File output: {output_gt}")
