import pandas as pd
import csv
import itertools
import time

def generate_candidate_pairs_B1(
    file_a,
    file_b,
    output_file,
    chunk_size=200_000
):
    """
    Genera candidate pairs secondo il blocking B1: stesso manufacturer e stesso year.
    - Tutti i campi sono trattati come stringhe (dtype=str)
    - Nessun filtro su 'invalid'
    - file_a entra tutto in RAM
    - file_b viene letto a chunk
    """

    print(f"📥 Caricamento file A in RAM: {file_a}")
    df_a = pd.read_csv(file_a, dtype=str)
    print(f"✔ Record totali in A: {len(df_a)}")

    first_chunk = True
    total_pairs = 0

    print("🚀 Inizio scansione file B a chunk...")

    for i, chunk_b in enumerate(
        pd.read_csv(file_b, chunksize=chunk_size, dtype=str)
    ):
        print(f"\n➡ Processo chunk {i+1} | righe lette: {len(chunk_b)}")

        if chunk_b.empty:
            print("↳ Chunk vuoto, salto.")
            continue

        # merge sul blocking key (stringhe)
        merged = pd.merge(
            df_a,
            chunk_b,
            on=["manufacturer", "year"],
            how="inner",
            suffixes=("_a", "_b")
        )

        num_pairs = len(merged)
        total_pairs += num_pairs

        print(f"↳ candidate pairs trovate in questo chunk: {num_pairs}")
        print(f"↳ totale candidate pairs finora: {total_pairs}")

        if num_pairs == 0:
            continue

        # colonne da scrivere (schema tipo ground truth)
        cols_a = [c + "_a" for c in df_a.columns if c not in ["manufacturer", "year"]]
        cols_b = [c + "_b" for c in chunk_b.columns if c not in ["manufacturer", "year"]]

        ordered_cols = ["manufacturer", "year"] + cols_a + cols_b

        merged[ordered_cols].to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk
        )

        first_chunk = False

    print("\n✅ Blocking B1 completato")
    print(f"📁 File output: {output_file}")
    print(f"✔ Totale candidate pairs generate: {total_pairs}")

def normalize_fuel_type_for_blocking(ft):
    """Normalizza i valori speciali di fuel_type per il blocking."""
    ft = str(ft).strip().lower()
    if ft == "flex fuel vehicle":
        return "gasoline"
    elif ft == "biodiesel":
        return "diesel"
    elif ft in ["compressed natural gas", "propane"]:
        return "other"
    else:
        return ft

def generate_candidate_pairs_B2(
    file_a,
    file_b,
    output_file,
    log_every=100_000   # stampa ogni N candidate pairs
):

    start_time = time.time()

    print("📥 Caricamento completo in RAM...")

    df_a = pd.read_csv(file_a, dtype=str)
    df_b = pd.read_csv(file_b, dtype=str)

    df_a["fuel_type"] = df_a["fuel_type"].apply(normalize_fuel_type_for_blocking)
    df_b["fuel_type"] = df_b["fuel_type"].apply(normalize_fuel_type_for_blocking)

    print(f"✔ Record A: {len(df_a)}")
    print(f"✔ Record B: {len(df_b)}")

    a_by_block = {
        key: g for key, g in df_a.groupby(["transmission", "year", "fuel_type"])
    }

    b_by_block = {
        key: g for key, g in df_b.groupby(["transmission", "year", "fuel_type"])
    }

    print("🚀 Inizio generazione candidate pairs (streaming)...")

    total_pairs = 0
    next_log = log_every

    with open(output_file, "w", newline="", encoding="utf-8") as f_out:
        writer = None

        for block_key, block_a in a_by_block.items():

            block_b = b_by_block.get(block_key)
            if block_b is None:
                continue

            for _, row_a in block_a.iterrows():
                for _, row_b in block_b.iterrows():

                    row_out = {}

                    for c in df_a.columns:
                        row_out[f"a_{c}"] = row_a[c]

                    for c in df_b.columns:
                        row_out[f"b_{c}"] = row_b[c]

                    if writer is None:
                        writer = csv.DictWriter(
                            f_out,
                            fieldnames=row_out.keys()
                        )
                        writer.writeheader()

                    writer.writerow(row_out)
                    total_pairs += 1

                    # 🔥 LOG PROGRESSIVO
                    if total_pairs >= next_log:
                        elapsed = time.time() - start_time
                        print(f"⏳ {total_pairs:,} candidate pairs scritte | "
                              f"{elapsed:.1f} sec trascorsi")
                        next_log += log_every

    elapsed_total = time.time() - start_time

    print("\n✅ Blocking B2 completato")
    print(f"✔ Totale candidate pairs generate: {total_pairs:,}")
    print(f"⏱ Tempo totale: {elapsed_total:.1f} secondi")


