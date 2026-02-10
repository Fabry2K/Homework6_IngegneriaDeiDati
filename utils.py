import pandas as pd
import math
import random
from collections import defaultdict
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

    for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str):
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


# ===============================
# CONFIGURAZIONE GENERALE
# ===============================
CHUNKSIZE = 100_000


# ==========================================================
# COUNT NULLS AND UNIQUES VALUES IN THE CSV FILE
# ==========================================================

# Impostazioni per stampa completa
pd.set_option("display.max_rows", None)        # stampa tutte le righe
pd.set_option("display.max_columns", None)     # stampa tutte le colonne
pd.set_option("display.width", None)           # nessun limite di larghezza
pd.set_option("display.max_colwidth", None)    # stampa completa del contenuto delle celle

def count_nulls_and_uniques(path, name, chunksize=50_000):
    null_counts = None
    unique_sets = defaultdict(set)
    total_rows = 0

    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str):
        if null_counts is None:
            null_counts = chunk.isnull().sum()
        else:
            null_counts += chunk.isnull().sum()

        for col in chunk.columns:
            unique_sets[col].update(chunk[col].dropna().unique())

        total_rows += len(chunk)

    report = pd.DataFrame({
        "null_count": null_counts,
        "null_%": (null_counts / total_rows) * 100,
        "unique_values": {col: len(vals) for col, vals in unique_sets.items()}
    })

    report["null_%"] = report["null_%"].round(2)
    report = report.sort_values(by="null_%", ascending=False)

    print("=" * 80)
    print(f"DATASET: {name}")
    print(f"Numero di record: {total_rows}")
    print(f"Numero di attributi: {len(report)}")
    print("=" * 80)
    print(report)

    return report


def desc_similarity_block(descriptions):
    """
    Calcola la matrice di similarità cosine tra le descrizioni.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )
    tfidf = vectorizer.fit_transform(descriptions)
    return cosine_similarity(tfidf)

def deduplicate_csv(
    csv_path,
    csv_out_clean,        # percorso del file pulito
    csv_out_duplicates,   # percorso del file dei duplicati
    desc_threshold=0.7
):
    import pandas as pd
    import hashlib
    import csv
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity


    # -------------------------
    # Similarità descrizioni
    # -------------------------
    def desc_similarity_block(descriptions):
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000
        )
        tfidf = vectorizer.fit_transform(descriptions)
        return cosine_similarity(tfidf)

    # -------------------------
    # Hash helper
    # -------------------------
    def hash_row(s):
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    # -------------------------
    # Load
    # -------------------------
    df = pd.read_csv(csv_path, dtype=str)
    initial_count = len(df)

    # -------------------------
    # Colonne da confrontare
    # -------------------------
    cols_to_check = [
        col for col in df.columns
        if col not in [
            "id", "url", "posting_date", "region", "region_url",
            "price", "odometer", "title_status", "size", "type",
            "image_url", "state", "lat", "long"
        ]
    ]
    cols_to_compare = [c for c in cols_to_check if c not in ["VIN", "description"]]

    # -------------------------
    # Colonne temporanee (NO modifica dati originali)
    # -------------------------
    df["_vin_norm"] = df["VIN"].fillna("__NULL_VIN__")
    df["_desc_norm"] = df["description"].fillna("")

    df["_base_str"] = (
        df[cols_to_compare]
        .fillna("__NULL__")
        .astype(str)
        .agg("|".join, axis=1)
    )

    df["_hash"] = df["_base_str"].apply(hash_row)

    # -------------------------
    # Deduplica
    # -------------------------
    to_drop = set()
    involved = set()
    processed = 0
    removed = 0
    total = len(df)

    for vin_value, vin_cluster in df.groupby("_vin_norm"):
        if len(vin_cluster) < 2:
            continue

        for _, hash_cluster in vin_cluster.groupby("_hash"):
            indices = list(hash_cluster.index)
            if len(indices) < 2:
                continue

            for i_idx in indices:
                if i_idx in to_drop:
                    continue

                rec_i = df.loc[i_idx]

                for j_idx in indices:
                    if j_idx <= i_idx or j_idx in to_drop:
                        continue

                    rec_j = df.loc[j_idx]

                    try:
                        sim = desc_similarity_block(
                            [rec_i["_desc_norm"], rec_j["_desc_norm"]]
                        )
                        desc_match = sim[0, 1] >= desc_threshold
                    except ValueError:
                        desc_match = rec_i["_desc_norm"] == rec_j["_desc_norm"]

                    if desc_match:
                        removed += 1
                        print(
                            f"Trovato doppione ({removed}) → "
                            f"cancello index={j_idx} (master={i_idx})"
                        )

                        to_drop.add(j_idx)
                        involved.update([i_idx, j_idx])

                processed += 1
                print(f"Record processati: {processed}/{total}")

    # -------------------------
    # Output finali
    # -------------------------
    df_clean = df.drop(index=list(to_drop))
    df_duplicates = df.loc[list(involved)]

    temp_cols = ["_vin_norm", "_desc_norm", "_base_str", "_hash"]
    df_clean = df_clean.drop(columns=temp_cols)
    df_duplicates = df_duplicates.drop(columns=temp_cols)

    # 🔐 FIX FONDAMENTALE: quoting corretto
    df_clean.to_csv(
        csv_out_clean,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\"
    )

    df_duplicates.to_csv(
        csv_out_duplicates,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\"
    )

    print("\nDeduplicazione completata")
    print(f"Record iniziali: {initial_count}")
    print(f"Record eliminati: {removed}")
    print(f"Record finali: {len(df_clean)}")

    return df_clean, df_duplicates, removed


# Extract records with invalid = 0
def extract_valid_records(input_csv, output_csv, chunksize=200_000):
    """
    Estrae tutti i record con 'invalid' == 0 e li scrive in un nuovo CSV.
    Lavora a chunk per file grandi.
    """
    first_chunk = True
    total_written = 0

    for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str):
        # Assicurati che la colonna 'invalid' esista
        if 'invalid' not in chunk.columns:
            raise ValueError("'invalid' column not found in CSV")

        # Filtra i record validi
        valid_chunk = chunk[chunk['invalid'] == 0]

        # Scrivi su CSV
        if not valid_chunk.empty:
            valid_chunk.to_csv(
                output_csv,
                mode='w' if first_chunk else 'a',
                index=False,
                header=first_chunk
            )
            first_chunk = False
            total_written += len(valid_chunk)

        print(f"Processato chunk di {len(chunk)} righe, scritti {len(valid_chunk)} validi")

    print(f"\nEstrazione completata: {total_written} record validi scritti in {output_csv}")



def count_unique_vins_in_memory(input_csv):
    """
    Conta quanti VIN nel CSV appaiono solo in un record (non duplicati)
    caricando tutto in memoria.
    """
    # Carica tutto in memoria
    df = pd.read_csv(input_csv, usecols=['vin'], dtype=str)

    # Rimuovi VIN vuoti o NaN
    df = df[df['vin'].notna() & (df['vin'].str.strip() != "")]

    # Conta quante volte appare ogni VIN
    vin_counts = df['vin'].value_counts()

    # Seleziona quelli che appaiono solo 1 volta
    unique_vins = vin_counts[vin_counts == 1]

    print(f"Totale VIN nel file: {len(vin_counts)}")
    print(f"VIN non duplicati (occurrence = 1): {len(unique_vins)}")

    return unique_vins.index.tolist()  # opzionale: ritorna la lista dei VIN unici


#Remove VIN from a normal csv file
def remove_vin_from_dataset(input_csv, output_csv, chunksize=200_000):
    """
    Rimuove la colonna 'vin' da un CSV e salva il risultato su output_csv.
    Lavora a chunk.
    """
    print(f"🧹 Rimozione campo VIN da {input_csv}...")

    first_chunk = True

    for chunk_idx, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunksize, dtype=str)):
        if "vin" in chunk.columns:
            chunk = chunk.drop(columns=["vin"])

        chunk.to_csv(
            output_csv,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk
        )

        first_chunk = False
        print(f"✔ Processato chunk {chunk_idx} ({len(chunk)} righe)")

    print(f"✅ File salvato senza VIN: {output_csv}")

#Remove VIN from ground truth file
def remove_vins_from_ground_truth(input_gt, output_gt, chunksize=200_000):
    """
    Rimuove i campi 'a_vin' e 'b_vin' dalla ground truth.
    Lavora a chunk.
    """
    print(f"🧹 Rimozione a_vin e b_vin da {input_gt}...")

    first_chunk = True

    for chunk_idx, chunk in enumerate(pd.read_csv(input_gt, chunksize=chunksize, dtype=str)):
        cols_to_drop = [c for c in ["a_vin", "b_vin"] if c in chunk.columns]
        if cols_to_drop:
            chunk = chunk.drop(columns=cols_to_drop)

        chunk.to_csv(
            output_gt,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk
        )

        first_chunk = False
        print(f"✔ Processato chunk {chunk_idx} ({len(chunk)} righe)")

    print(f"✅ Ground truth senza VIN salvata in: {output_gt}")

def split_ground_truth(
    input_gt,
    train_out,
    val_out,
    test_out,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    """
    Divide la ground truth in train / validation / test.

    Split default: 70% train, 20% validation, 10% test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Le percentuali devono sommare a 1"

    print(f"📥 Caricamento ground truth da {input_gt}...")
    df = pd.read_csv(input_gt, dtype=str)

    total = len(df)
    print(f"📊 Record totali: {total}")

    # shuffle riproducibile
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    # scrittura file
    df_train.to_csv(train_out, index=False)
    df_val.to_csv(val_out, index=False)
    df_test.to_csv(test_out, index=False)

    print("✅ Split completato:")
    print(f"  🟢 Train:      {len(df_train)} ({len(df_train)/total:.1%})")
    print(f"  🟡 Validation: {len(df_val)} ({len(df_val)/total:.1%})")
    print(f"  🔵 Test:       {len(df_test)} ({len(df_test)/total:.1%})")

    print("\n📁 File generati:")
    print(f"  - {train_out}")
    print(f"  - {val_out}")
    print(f"  - {test_out}")


import csv
from itertools import islice

def stampa_prime_10_righe(file_csv):
    """
    Legge un file CSV e stampa solo le prime 10 righe senza caricare tutto il file in memoria.
    """
    with open(file_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in islice(reader, 10):  # prende solo le prime 10 righe
            print(row)


