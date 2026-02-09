import pandas as pd

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
