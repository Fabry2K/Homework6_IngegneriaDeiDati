import csv
import os
import dedupe
from dedupe import variables

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
    elif (value_1 == 'truck' and value_2 == 'pickup') or (value_1 == 'pickup' and value_2 == 'truck'):
        return 0
    elif (value_1 == 'offroad' and value_2 == 'suv') or (value_1 == 'suv' and value_2 == 'offroad'):
        return 0
    elif (value_1 == 'pickup' and value_2 == 'offroad') or (value_1 == 'offroad' and value_2 == 'pickup'):
        return 0
    else:
        return 1

def drive_matcher(value_1, value_2):
    if (value_1 == '4wd' and value_2 == 'awd') or (value_1 == 'awd' and value_2 == '4wd'):
        return 0
    elif (value_1 == 'fwd' and value_2 == '4x2') or (value_1 == '4x2' and value_2 == 'fwd'):
        return 0
    elif (value_1 == 'rwd' and value_2 == '4x2') or (value_1 == '4x2' and value_2 == 'rwd'):
        return 0
    else:
        return 1

# ######################
# # READ DATA FUNCTION #
# ######################

def readData(filename):
    print("[readData] Inizio lettura CSV:", filename)

    data_A = {}
    data_B = {}

    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            record_A = {
                k[2:]: (to_float(v) if k[2:] == "mileage" else (v if v != "" else None))
                for k, v in row.items() if k.startswith("a_")
            }
            data_A[f"A_{i}"] = record_A

            record_B = {
                k[2:]: (to_float(v) if k[2:] == "mileage" else (v if v != "" else None))
                for k, v in row.items() if k.startswith("b_")
            }
            data_B[f"B_{i}"] = record_B


    print(f"[readData] Record A: {len(data_A)}")
    print(f"[readData] Record B: {len(data_B)}")

    return data_A, data_B

# ######################
# # DEDUPE FUNCTION    #
# ######################

def dedupe_labels(filename, training_json):
    print("\n[dedupe_labels] ===== INIZIO =====")

    if not os.path.exists(training_json):
        raise FileNotFoundError(f"File di training JSON NON trovato: {training_json}")

    print("[dedupe_labels] File JSON trovato:", training_json)

    data_A, data_B = readData(filename)

    print("[dedupe_labels] Definizione campi Dedupe")

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

    print("[dedupe_labels] Apro file JSON di training")
    with open(training_json, "r", encoding="utf-8") as tf:
        print("[dedupe_labels] Avvio prepare_training()")
        linker.prepare_training(
            data_A,
            data_B,
            training_file=tf
        )

    print("[dedupe_labels] prepare_training COMPLETATO")

    print("[dedupe_labels] >>> INIZIO TRAINING <<<")
    linker.train()
    print("[dedupe_labels] >>> FINE TRAINING <<<")

    print("[dedupe_labels] ===== FINE =====\n")
    return linker