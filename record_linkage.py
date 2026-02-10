import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

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
# Funzione di valutazione con chiave univoca completa
# ------------------------------

def evaluate_B1(blocking_file, test_file, chunk_size=200000, match_threshold=0.70):
    # Carico test file
    df_test = pd.read_csv(test_file, dtype=str)
    
    # Creo dizionario di lookup: chiave = tutti i campi principali di A e B, valore = match
    test_dict = {
        (
            safe_str(df_test.loc[i, 'a_manufacturer']), safe_str(df_test.loc[i, 'a_model']), safe_str(df_test.loc[i, 'a_year']),
            safe_str(df_test.loc[i, 'a_mileage']), safe_str(df_test.loc[i, 'a_fuel_type']), safe_str(df_test.loc[i, 'a_transmission']),
            safe_str(df_test.loc[i, 'a_body_type']), safe_str(df_test.loc[i, 'a_cylinders']), safe_str(df_test.loc[i, 'a_drive']),
            safe_str(df_test.loc[i, 'a_color']),
            safe_str(df_test.loc[i, 'b_manufacturer']), safe_str(df_test.loc[i, 'b_model']), safe_str(df_test.loc[i, 'b_year']),
            safe_str(df_test.loc[i, 'b_mileage']), safe_str(df_test.loc[i, 'b_fuel_type']), safe_str(df_test.loc[i, 'b_transmission']),
            safe_str(df_test.loc[i, 'b_body_type']), safe_str(df_test.loc[i, 'b_cylinders']), safe_str(df_test.loc[i, 'b_drive']),
            safe_str(df_test.loc[i, 'b_color'])
        ): int(df_test.loc[i, 'match'])
        for i in df_test.index
    }
    
    y_true = []
    y_pred = []
    
    start_train = time.time()
    end_train = time.time()
    start_infer = time.time()
    
    chunk_number = 0
    total_rows = 0
    matched_in_test = 0
    
    for chunk in pd.read_csv(blocking_file, chunksize=chunk_size, dtype=str):
        chunk_number += 1
        print(f"\n--- Elaborazione chunk {chunk_number} ---")
        for idx, row in chunk.iterrows():
            total_rows += 1
            
            # Calcolo punteggio totale
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
            
            # Creo chiave univoca della coppia (A+B)
            pair_tuple = (
                safe_str(row['manufacturer']), safe_str(row['model_a']), safe_str(row['year']),
                safe_str(row['mileage_a']), safe_str(row['fuel_type_a']), safe_str(row['transmission_a']),
                safe_str(row['body_type_a']), safe_str(row['cylinders_a']), safe_str(row['drive_a']),
                safe_str(row['color_a']),
                safe_str(row.get('manufacturer_b', row['manufacturer'])),  # se non c'è manufacturer_b, uso lo stesso
                safe_str(row['model_b']), safe_str(row.get('year_b', row['year'])),
                safe_str(row.get('mileage_b', row['mileage_a'])), safe_str(row.get('fuel_type_b', row['fuel_type_a'])),
                safe_str(row.get('transmission_b', row['transmission_a'])), safe_str(row.get('body_type_b', row['body_type_a'])),
                safe_str(row.get('cylinders_b', row['cylinders_a'])), safe_str(row.get('drive_b', row['drive_a'])),
                safe_str(row.get('color_b', row['color_a']))
            )
            
            if pair_tuple in test_dict:
                matched_in_test += 1
                true_match = test_dict[pair_tuple]
                y_true.append(true_match)
                y_pred.append(pred_match)
            
            if total_rows % 1000 == 0:
                print(f"Righe processate: {total_rows}, coppie trovate nel test set: {matched_in_test}")
        
        print(f"Chunk {chunk_number} completato. Righe processate finora: {total_rows}")
    
    end_infer = time.time()
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    train_time = end_train - start_train
    infer_time = end_infer - start_infer
    
    print("\n--- Elaborazione completata ---")
    print(f"Righe totali processate: {total_rows}")
    print(f"Coppie trovate nel test set: {matched_in_test}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Tempo train: {train_time:.2f}s, Tempo inferenza: {infer_time:.2f}s")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time,
        'inference_time': infer_time
    }
