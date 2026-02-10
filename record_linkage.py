import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock
from threading import Lock
import csv

# ------------------------------
# Funzioni di scoring
# ------------------------------
def safe_str(val):
    if val is None:
        return ''
    try:
        val_str = str(val).strip()
        if val_str.lower() == 'nan':
            return ''
        return val_str
    except:
        return ''

def score_model(model_a, model_b, max_score=0.5):
    model_a_str = safe_str(model_a)
    model_b_str = safe_str(model_b)
    if model_a_str == '' or model_b_str == '':
        return 0
    vectorizer = TfidfVectorizer().fit([model_a_str, model_b_str])
    v = vectorizer.transform([model_a_str, model_b_str])
    cos_sim = cosine_similarity(v[0], v[1])[0][0]
    return cos_sim * max_score

def score_mileage(mileage_a, mileage_b, max_score=0.1, max_diff=50000):
    if mileage_a is None or mileage_b is None:
        return 0
    try:
        diff = abs(int(float(mileage_a)) - int(float(mileage_b)))
    except:
        return 0
    if diff > max_diff:
        return 0
    return max_score * (1 - diff / max_diff)

def score_fuel(fuel_a, fuel_b, max_score=0.05):
    fuel_a_str = safe_str(fuel_a).lower()
    fuel_b_str = safe_str(fuel_b).lower()
    if fuel_a_str == '' or fuel_b_str == '':
        return 0
    equivalences = {'flex fuel vehicle': 'gasoline', 'biodiesel': 'diesel'}
    if fuel_a_str == fuel_b_str:
        return max_score
    elif fuel_a_str in equivalences and equivalences[fuel_a_str] == fuel_b_str:
        return max_score * 0.8
    elif fuel_b_str in equivalences and equivalences[fuel_b_str] == fuel_a_str:
        return max_score * 0.8
    elif fuel_a_str == 'other' or fuel_b_str == 'other':
        return max_score * 0.3
    return 0

def score_body(body_a, body_b, max_score=0.05):
    body_a_str = safe_str(body_a).lower()
    body_b_str = safe_str(body_b).lower()
    if body_a_str == '' or body_b_str == '':
        return 0
    equivalences = {'truck': 'pickup', 'offroad': ['suv','pickup']}
    if body_a_str == body_b_str:
        return max_score
    elif body_a_str in equivalences:
        if isinstance(equivalences[body_a_str], list):
            if body_b_str in equivalences[body_a_str]:
                return max_score * 0.8
        else:
            if body_b_str == equivalences[body_a_str]:
                return max_score * 0.8
    elif body_b_str in equivalences:
        if isinstance(equivalences[body_b_str], list):
            if body_a_str in equivalences[body_b_str]:
                return max_score * 0.8
        else:
            if body_a_str == equivalences[body_b_str]:
                return max_score * 0.8
    elif body_a_str == 'other' or body_b_str == 'other':
        return max_score * 0.3
    return 0

def score_cylinders(cyl_a, cyl_b, max_score=0.03):
    cyl_a_str = safe_str(cyl_a).lower()
    cyl_b_str = safe_str(cyl_b).lower()
    if cyl_a_str == '' or cyl_b_str == '':
        return 0
    if cyl_a_str == cyl_b_str:
        return max_score
    elif cyl_a_str == 'other' or cyl_b_str == 'other':
        return max_score * 0.3
    return 0

def score_drive(drive_a, drive_b, max_score=0.02):
    drive_a_str = safe_str(drive_a).lower()
    drive_b_str = safe_str(drive_b).lower()
    if drive_a_str == '' or drive_b_str == '':
        return 0
    equivalences = {'4wd':'awd','awd':'4wd','fwd':'4x2','rwd':'4x2'}
    if drive_a_str == drive_b_str:
        return max_score
    elif drive_a_str in equivalences and equivalences[drive_a_str] == drive_b_str:
        return max_score * 0.8
    elif drive_b_str in equivalences and equivalences[drive_b_str] == drive_a_str:
        return max_score * 0.8
    return 0

def score_color(color_a, color_b, max_score=0.02):
    color_a_str = safe_str(color_a).lower()
    color_b_str = safe_str(color_b).lower()
    if color_a_str == '' or color_b_str == '':
        return 0
    return max_score if color_a_str == color_b_str else 0

def score_exact(field_a, field_b, max_score):
    field_a_str = safe_str(field_a)
    field_b_str = safe_str(field_b)
    if field_a_str == '' or field_b_str == '':
        return 0
    return max_score if field_a_str == field_b_str else 0

# ------------------------------
# Funzione ottimizzata con early pruning e print su test set
# ------------------------------
def evaluate_B1(blocking_file, test_file, chunk_size=1000000, match_threshold=0.70):
    # Carico test file
    df_test = pd.read_csv(test_file, dtype=str)
    
    # Creo dizionario di lookup: chiave = tutti i campi principali di A e B, valore = match
    test_dict = {
        tuple(safe_str(df_test.loc[i, col]) for col in [
            'a_manufacturer','a_model','a_year','a_mileage','a_fuel_type','a_transmission','a_body_type',
            'a_cylinders','a_drive','a_color',
            'b_manufacturer','b_model','b_year','b_mileage','b_fuel_type','b_transmission','b_body_type',
            'b_cylinders','b_drive','b_color'
        ]): int(df_test.loc[i, 'match'])
        for i in df_test.index
    }

    # Traccia righe test set valutate
    evaluated_test_set = set()
    
    y_true = []
    y_pred = []
    
    start_train = time.time()
    end_train = time.time()
    start_infer = time.time()
    
    chunk_number = 0
    total_rows = 0
    
    for chunk in pd.read_csv(blocking_file, chunksize=chunk_size, dtype=str):
        chunk_number += 1
        print(f"\n--- Elaborazione chunk {chunk_number} ---")
        for idx, row in chunk.iterrows():
            total_rows += 1
            
            # Creo chiave univoca della coppia
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
            
            # Early pruning: calcolo solo se la coppia è nel test set
            if pair_tuple in test_dict:
                true_match = test_dict[pair_tuple]
                
                # Calcolo punteggio
                total_score = 0
                total_score += score_model(row['model_a'], row['model_b'])
                total_score += score_exact(row['manufacturer'], row['manufacturer'], 0.2)
                total_score += score_exact(row['year'], row['year'], 0.1)
                total_score += score_mileage(row['mileage_a'], row['mileage_b'])
                total_score += score_fuel(row['fuel_type_a'], row['fuel_type_b'])
                total_score += score_exact(row['transmission_a'], row['transmission_b'], 0.05)
                total_score += score_body(row['body_type_a'], row['body_type_b'])
                total_score += score_cylinders(row['cylinders_a'], row['cylinders_b'])
                total_score += score_drive(row['drive_a'], row['drive_b'])
                total_score += score_color(row['color_a'], row['color_b'])
                
                pred_match = 1 if total_score >= match_threshold else 0
                y_true.append(true_match)
                y_pred.append(pred_match)
                
                evaluated_test_set.add(pair_tuple)
                
                if len(evaluated_test_set) % 100 == 0:
                    print(f"Righe del test set già valutate: {len(evaluated_test_set)} / {len(df_test)}")
        
        print(f"Chunk {chunk_number} completato. Test set valutato finora: {len(evaluated_test_set)} / {len(df_test)}")
    
    end_infer = time.time()
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    train_time = end_train - start_train
    infer_time = end_infer - start_infer
    
    print("\n--- Elaborazione completata ---")
    print(f"Righe totali processate dal blocking file: {total_rows}")
    print(f"Righe del test set valutate: {len(evaluated_test_set)} / {len(df_test)}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Tempo train: {train_time:.2f}s, Tempo inferenza: {infer_time:.2f}s")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time,
        'inference_time': infer_time
    }


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
def process_chunk(chunk, test_dict, match_threshold=0.70, backup_file=None):
    y_true_chunk = []
    y_pred_chunk = []
    evaluated_test_set_chunk = set()

    # Apriamo il file di backup in append (con lock)
    backup_fp = None
    if backup_file:
        backup_fp = open(backup_file, 'a', newline='', encoding='utf-8')
        backup_writer = csv.writer(backup_fp)
    
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
            # calcolo punteggio (qui puoi usare le tue regole)
            total_score = 0
            total_score += score_model(row['model_a'], row['model_b'])
            total_score += score_exact(row['manufacturer'], row['manufacturer'], 0.2)
            total_score += score_exact(row['year'], row['year'], 0.1)
            total_score += score_mileage(row['mileage_a'], row['mileage_b'])
            total_score += score_fuel(row['fuel_type_a'], row['fuel_type_b'])
            total_score += score_exact(row['transmission_a'], row['transmission_b'], 0.05)
            total_score += score_body(row['body_type_a'], row['body_type_b'])
            total_score += score_cylinders(row['cylinders_a'], row['cylinders_b'])
            total_score += score_drive(row['drive_a'], row['drive_b'])
            total_score += score_color(row['color_a'], row['color_b'])

            pred_match = 1 if total_score >= match_threshold else 0
            true_match = test_dict[pair_tuple]

            y_true_chunk.append(true_match)
            y_pred_chunk.append(pred_match)
            evaluated_test_set_chunk.add(pair_tuple)

            # Scrittura backup (thread-safe)
            if backup_file:
                with write_lock:
                    backup_writer.writerow([pair_tuple, pred_match, true_match])

    if backup_fp:
        backup_fp.close()

    return y_true_chunk, y_pred_chunk, evaluated_test_set_chunk


# ------------------------------
# Funzione principale Windows-safe + Ctrl+C
# ------------------------------
def evaluate_B1_parallel(blocking_file, test_file, chunk_size=500_000, match_threshold=0.70, max_workers=8, backup_file=None):
    # Carico test set
    df_test = pd.read_csv(test_file, dtype=str)
    test_dict = {
        tuple(safe_str(df_test.loc[i, col]) for col in [
            'a_manufacturer','a_model','a_year','a_mileage','a_fuel_type','a_transmission','a_body_type',
            'a_cylinders','a_drive','a_color',
            'b_manufacturer','b_model','b_year','b_mileage','b_fuel_type','b_transmission','b_body_type',
            'b_cylinders','b_drive','b_color'
        ]): int(df_test.loc[i, 'match'])
        for i in df_test.index
    }

    y_true = []
    y_pred = []
    evaluated_test_set = set()

    start_time = time.time()

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            chunk_number = 0
            chunk_iter = pd.read_csv(blocking_file, chunksize=chunk_size, dtype=str)

            for chunk in chunk_iter:
                chunk_number += 1
                print(f"\nInvio chunk {chunk_number} ai worker...")
                future = executor.submit(process_chunk, chunk, test_dict, match_threshold, backup_file)
                futures.add(future)

                # Limita chunk in RAM a max_workers
                while len(futures) >= max_workers:
                    done, futures = wait_one_done(futures)
                    for y_true_chunk, y_pred_chunk, evaluated_chunk in done:
                        y_true.extend(y_true_chunk)
                        y_pred.extend(y_pred_chunk)
                        evaluated_test_set.update(evaluated_chunk)
                        print(f"Test set valutato finora: {len(evaluated_test_set)} / {len(test_dict)}")

            # Elabora i chunk rimanenti
            for future in as_completed(futures):
                y_true_chunk, y_pred_chunk, evaluated_chunk = future.result()
                y_true.extend(y_true_chunk)
                y_pred.extend(y_pred_chunk)
                evaluated_test_set.update(evaluated_chunk)

    except KeyboardInterrupt:
        print("\nInterruzione ricevuta! Terminazione immediata dei worker...")
        executor.shutdown(wait=False)
        return

    # ------------------------------
    # Gestione delle righe del test set non trovate nel blocking
    # ------------------------------
    missing_pairs = set(test_dict.keys()) - evaluated_test_set
    for pair in missing_pairs:
        y_true.append(test_dict[pair])
        y_pred.append(0)

        # Scrittura backup
        if backup_file:
            with write_lock:
                with open(backup_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([pair, 0, test_dict[pair]])

    end_time = time.time()

    # Calcolo metriche
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n--- Elaborazione completata ---")
    print(f"Righe del test set valutate: {len(evaluated_test_set)} / {len(test_dict)}")
    print(f"Totale coppie (incluse quelle non trovate nel blocking): {len(y_true)}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Tempo totale: {end_time - start_time:.2f}s")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_time': end_time - start_time
    }


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