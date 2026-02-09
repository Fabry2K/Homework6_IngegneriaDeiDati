import pandas as pd

def generate_candidate_pairs_B1(file_a, file_b, output_file, chunk_size=200_000):
    """
    Genera candidate pairs secondo il blocking B1: stesso manufacturer e stesso year.
    - file_a: CSV che entra tutto in RAM
    - file_b: CSV processato a chunk
    - output_file: CSV dove salvare le candidate pairs
    - chunk_size: dimensione dei chunk di file_b
    """
    print(f"Carico il primo file ({file_a}) in RAM...")
    df_a = pd.read_csv(file_a)
    df_a = df_a[df_a['invalid'] == 0]  # consideriamo solo record validi
    print(f"Record validi in file A: {len(df_a)}")

    first_chunk = True
    total_pairs = 0

    for i, chunk_b in enumerate(pd.read_csv(file_b, chunksize=chunk_size)):
        print(f"\nProcesso chunk {i+1} di {chunk_b.shape[0]} righe da file B...")

        chunk_b = chunk_b[chunk_b['invalid'] == 0]  # solo record validi

        # merge sul blocking key
        merged = pd.merge(
            df_a,
            chunk_b,
            on=['manufacturer', 'year'],
            how='inner',
            suffixes=('_a', '_b')
        )

        num_pairs = len(merged)
        total_pairs += num_pairs
        print(f"↳ candidate pairs trovate in questo chunk: {num_pairs}")
        print(f"↳ totale candidate pairs finora: {total_pairs}")

        if num_pairs == 0:
            continue

        # colonne da scrivere: chiavi + resto con suffissi
        cols_a = [c+'_a' for c in df_a.columns if c not in ['invalid', 'manufacturer', 'year']]
        cols_b = [c+'_b' for c in df_a.columns if c not in ['invalid', 'manufacturer', 'year']]
        ordered_cols = ['manufacturer', 'year'] + cols_a + cols_b

        merged[ordered_cols].to_csv(
            output_file,
            mode='w' if first_chunk else 'a',
            index=False,
            header=first_chunk
        )
        first_chunk = False

    print(f"\nCandidate pairs generate e salvate in {output_file}")
    print(f"Totale candidate pairs generate: {total_pairs}")
