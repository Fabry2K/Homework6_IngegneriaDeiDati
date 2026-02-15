import csv
import dedupe
from dedupe import variables
import random
import json

######################
# UTILITY FUNCTIONS  #
######################

def to_float(x):
    try:
        return float(x)
    except:
        return None

def other_as_match(value_1, value_2):
    if value_1 == value_2 == 'other':
        return 0
    elif value_1 == 'other' or value_2 == 'other':
        return 0
    else:
        return 1

def body_matcher(value_1, value_2):
    if other_as_match(value_1, value_2) == 0:
        return 0
    elif (value_1=='truck' and value_2=='pickup') or (value_1=='pickup' and value_2=='truck'):
        return 0
    elif (value_1=='offroad' and value_2=='suv') or (value_1=='suv' and value_2=='offroad'):
        return 0
    elif (value_1=='pickup' and value_2=='offroad') or (value_1=='offroad' and value_2=='pickup'):
        return 0
    else:
        return 1

def drive_matcher(value_1, value_2):
    if (value_1=='4wd' and value_2=='awd') or (value_1=='awd' and value_2=='4wd'):
        return 0
    elif (value_1=='fwd' and value_2=='4x2') or (value_1=='4x2' and value_2=='fwd'):
        return 0
    elif (value_1=='rwd' and value_2=='4x2') or (value_1=='4x2' and value_2=='rwd'):
        return 0
    else:
        return 1

######################
# READ DATA FUNCTION #
######################

def readData(filename, sample_size=None, seed=42):
    all_rows = []

    # Leggi tutto
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)

    # Campionamento
    if sample_size is not None and sample_size < len(all_rows):
        random.seed(seed)
        all_rows = random.sample(all_rows, sample_size)

    data_A = {}
    data_B = {}
    labeled_pairs = {"match": [], "distinct": []}

    for i, row in enumerate(all_rows):
        record_A = {
            k[2:]: (to_float(v) if k[2:] == "mileage" else (v if v != "" else None))
            for k, v in row.items() if k.startswith("a_")
        }
        record_B = {
            k[2:]: (to_float(v) if k[2:] == "mileage" else (v if v != "" else None))
            for k, v in row.items() if k.startswith("b_")
        }

        data_A[f"A_{i}"] = record_A
        data_B[f"B_{i}"] = record_B

        match_val = row['match'].strip() == '1'
        if match_val:
            labeled_pairs["match"].append((record_A, record_B))
        else:
            labeled_pairs["distinct"].append((record_A, record_B))

    return data_A, data_B, labeled_pairs

######################
# DEDUPE TRAINING    #
######################

def dedupe_labels(filename, sample_size=1000, settings_file="settings.json"):
    data_A, data_B, labeled_pairs = readData(filename, sample_size=sample_size)

    print("[DEBUG] Esempio record A:", list(data_A.values())[0])
    print("[DEBUG] Esempio record B:", list(data_B.values())[0])

    print(f"[dedupe_labels] Coppie match: {len(labeled_pairs['match'])}")
    print(f"[dedupe_labels] Coppie distinct: {len(labeled_pairs['distinct'])}")

    fields = [
        variables.String('manufacturer', has_missing=True),
        variables.Text('model', has_missing=True),
        variables.Exact('year', has_missing=True),
        variables.Price('mileage', has_missing=True),
        variables.Custom('fuel_type', comparator=other_as_match, has_missing=True),
        variables.String('transmission', has_missing=True),
        variables.Custom('body_type', comparator=body_matcher, has_missing=True),
        variables.Custom('cylinders', comparator=other_as_match, has_missing=True),
        variables.Custom('drive', comparator=drive_matcher, has_missing=True),
        variables.Text('color', has_missing=True)
    ]

    print("[dedupe_labels] Creo RecordLink")
    linker = dedupe.RecordLink(fields)

    print("[dedupe_labels] Avvio prepare_training")
    linker.prepare_training(data_A, data_B)

    print("[dedupe_labels] Passo le coppie etichettate")
    linker.mark_pairs(labeled_pairs)

    print("[dedupe_labels] Training in corso...")
    linker.train()
    print("[dedupe_labels] Training completato ✅")

    # 🔹 SALVATAGGIO AUTOMATICO DEL MODELLO
    with open(settings_file, "wb") as sf:
        linker.write_settings(sf)
    print(f"[dedupe_labels] Settings salvati in {settings_file}")

    return linker
