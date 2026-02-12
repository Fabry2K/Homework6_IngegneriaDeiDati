import pandas as pd
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock

# ------------------------------
# Funzione di sicurezza: converte valori in stringa
# ------------------------------
def safe_str(val):
    return '' if pd.isna(val) else str(val)

# ------------------------------
# Lock globale per la scrittura concorrente
# ------------------------------
write_lock = Lock()

# ------------------------------
# Worker per chunk
# ------------------------------
def process_chunk(chunk, test_dict, output_file=None):
    """Elabora un chunk e salva solo le righe che corrispondono al test set."""
    filtered_rows = []

    for idx, row in chunk.iterrows():
        # chiave univoca test set
        pair_tuple = (
            safe_str(row['manufacturer']), safe_str(row['model_a']), safe_str(row['year']),
            safe_str(row['mileage_a']), safe_str(row['fuel_type_a']), safe_str(row['transmission_a']),
            safe_str(row['body_type_a']), safe_str(row['cylinders_a']), safe_str(row['drive_a']),
            safe_str(row['color_a']),
            safe_str(row.get('manufacturer_b', row['manufacturer'])), safe_str(row['model_b']),
            safe_str(row.get('year_b', row['year'])), safe_str(row.get('mileage_b', row['mileage_a'])),
            safe_str(row.get('fuel_type_b', row['fuel_type_a'])), safe_str(row.get('transmission_b', row['transmission_a'])),
            safe_str(row.get('body_type_b', row['body_type_a'])), safe_str(row.get('cylinders_b', row['cylinders_a'])),
            safe_str(row.get('drive_b', row['drive_a'])), safe_str(row.get('color_b', row['color_a']))
        )

        if pair_tuple in test_dict:
            # Salva la riga originale nel file di backup
            filtered_rows.append(row)

    # Scrittura thread-safe su file
    if output_file and filtered_rows:
        with write_lock:
            with open(output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in filtered_rows:
                    writer.writerow(row.tolist())

    return len(filtered_rows)  # ritorna quante righe sono state salvate

# ------------------------------
# Funzione principale Windows-safe + Ctrl+C
# ------------------------------
def filter_candidate_pairs(blocking_file, test_file, output_file,
                                    chunk_size=500_000, max_workers=8):
    # Carico test set
    df_test = pd.read_csv(test_file, dtype=str)
    test_dict = {
        tuple(safe_str(df_test.loc[i, col]) for col in [
            'a_manufacturer','a_model','a_year','a_mileage','a_fuel_type','a_transmission','a_body_type',
            'a_cylinders','a_drive','a_color',
            'b_manufacturer','b_model','b_year','b_mileage','b_fuel_type','b_transmission','b_body_type',
            'b_cylinders','b_drive','b_color'
        ]): True
        for i in df_test.index
    }

    # Pulisce eventuale file di output precedente
    open(output_file, 'w').close()

    start_time = time.time()
    total_saved = 0

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            chunk_number = 0
            chunk_iter = pd.read_csv(blocking_file, chunksize=chunk_size, dtype=str)

            for chunk in chunk_iter:
                chunk_number += 1
                print(f"\nInvio chunk {chunk_number} al worker disponibile...")
                future = executor.submit(process_chunk, chunk, test_dict, output_file)
                futures.add(future)

                # Limita chunk in RAM a max_workers: aspetta che almeno uno finisca
                while len(futures) >= max_workers:
                    done, futures = wait_one_done(futures)
                    for saved_count in done:
                        total_saved += saved_count
                        print(f"Righe salvate finora: {total_saved}")

            # Elabora eventuali chunk rimanenti
            for future in as_completed(futures):
                saved_count = future.result()
                total_saved += saved_count
                print(f"Righe salvate finora: {total_saved}")

    except KeyboardInterrupt:
        print("\nInterruzione ricevuta! Terminazione immediata dei worker...")
        executor.shutdown(wait=False)
        return

    end_time = time.time()
    print("\n--- Elaborazione completata ---")
    print(f"Totale righe salvate nel file di backup: {total_saved}")
    print(f"Tempo totale: {end_time - start_time:.2f}s")

# ------------------------------
# Funzione helper per wait dei futures
# ------------------------------
def wait_one_done(futures_set):
    done = []
    for future in as_completed(futures_set):
        done.append(future.result())
        futures_set.remove(future)
        break
    return done, futures_set
