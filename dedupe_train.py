import csv
import json
import logging
import dedupe
from dedupe import variables
from sklearn.linear_model import LogisticRegression
import random

######################
# LOGGING SETUP
######################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

######################
# UTILITY FUNCTIONS
######################
def to_float(x):
    try:
        return float(x)
    except:
        return None

def other_as_match(v1, v2):
    if v1 is None or v2 is None:
        return None
    if v1 == v2 == "other":
        return 0
    if v1 == "other" or v2 == "other":
        return 0.5
    return 1

def body_matcher(v1, v2):
    if v1 is None or v2 is None:
        return None
    if v1 == v2:
        return 1
    equivalences = {
        ("truck", "pickup"),
        ("pickup", "truck"),
        ("offroad", "suv"),
        ("suv", "offroad"),
    }
    if (v1, v2) in equivalences:
        return 0
    return 0

def drive_matcher(v1, v2):
    if v1 is None or v2 is None:
        return None
    equivalences = {
        ("4wd", "awd"),
        ("awd", "4wd"),
        ("fwd", "4x2"),
        ("4x2", "fwd"),
        ("rwd", "4x2"),
        ("4x2", "rwd"),
    }
    if v1 == v2:
        return 1
    if (v1, v2) in equivalences:
        return 0
    return 0

######################
# READ DATASET
######################
def read_dataset(filename):
    print(f"[read_dataset] Caricamento CSV: {filename}")
    data = {}
    numeric_fields = {"year", "mileage", "price"}

    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            record = {}
            for k, v in row.items():
                if v is None or v.strip() == "":
                    record[k] = None
                elif k in numeric_fields:
                    record[k] = to_float(v)
                else:
                    record[k] = v.strip().lower()
            # uso l'id reale se esiste, altrimenti indice
            record_id = row.get("id", str(i))
            data[str(record_id)] = record

    print(f"[read_dataset] Record caricati: {len(data)}")
    return data

######################
# SAMPLE DICTIONARY
######################
def sample_dict(d, n, seed=42):
    random.seed(seed)
    keys = random.sample(list(d.keys()), min(n, len(d)))
    return {k: d[k] for k in keys}

######################
# GENERATE TRAINING JSON
######################
def generate_training_json(gt_file, data_A, data_B, json_file="training.json"):
    """
    gt_file: CSV con colonna 'match' (1=match, 0=distinct)
    Assumiamo che le righe corrispondano rispettivamente a data_A e data_B campionati
    """
    print(f"[generate_training_json] Generazione training JSON da {gt_file}")
    training = {"match": [], "distinct": []}

    # otteniamo le chiavi dei dict campionati
    keys_A = list(data_A.keys())
    keys_B = list(data_B.keys())

    with open(gt_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # usiamo modulo per non uscire dal numero di record campionati
            a_id = keys_A[i % len(keys_A)]
            b_id = keys_B[i % len(keys_B)]

            recA = data_A[a_id]
            recB = data_B[b_id]

            if row["match"] in ("1", 1):
                training["match"].append([recA, recB])
            else:
                training["distinct"].append([recA, recB])

    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(training, jf)

    print(f"[generate_training_json] JSON salvato in {json_file}")
    return json_file

######################
# DEDUPE TRAINING
######################
def dedupe_labels(datafile_a, datafile_b, groundtruth_file, sample_size=1000):
    print("\n[dedupe_labels] ===== INIZIO =====")

    data_A = read_dataset(datafile_a)
    data_B = read_dataset(datafile_b)

    # 🔹 CAMPIONAMENTO
    data_A_sample = sample_dict(data_A, sample_size)
    data_B_sample = sample_dict(data_B, sample_size)
    print(f"[dedupe_labels] Campionati {len(data_A_sample)} record da A e {len(data_B_sample)} da B")

    # 🔹 GENERA TRAINING JSON
    training_json = generate_training_json(
        groundtruth_file,
        data_A_sample,
        data_B_sample,
        "training.json"
    )

    # 🔹 DEFINIZIONE CAMPI
    fields = [
        variables.String("manufacturer", has_missing=True),
        variables.String("model", has_missing=True),
        variables.Exact("year", has_missing=True),
        variables.Price("mileage", has_missing=True),
        variables.Custom("fuel_type", comparator=other_as_match, has_missing=True),
        variables.String("transmission", has_missing=True),
        variables.Custom("body_type", comparator=body_matcher, has_missing=True),
        variables.Custom("cylinders", comparator=other_as_match, has_missing=True),
        variables.Custom("drive", comparator=drive_matcher, has_missing=True),
        variables.String("color", has_missing=True),
    ]

    # 🔹 CREAZIONE LINKER
    print("[dedupe_labels] Creo RecordLink")
    linker = dedupe.RecordLink(fields, num_cores=4)

    # 🔹 LOGISTIC REGRESSION
    linker.classifier = LogisticRegression(
        max_iter=500,
        solver="lbfgs"
    )

    # 🔹 PREPARE TRAINING
    print("[dedupe_labels] prepare_training()")
    with open(training_json, "r", encoding="utf-8") as tf:
        linker.prepare_training(
            data_A_sample,
            data_B_sample,
            training_file=tf,
            sample_size=5000
        )
    print("[dedupe_labels] prepare_training COMPLETATO")

    # 🔹 TRAINING
    logging.info(">>> INIZIO TRAINING <<<")
    linker.train(recall=0.8, index_predicates=False)
    logging.info(">>> FINE TRAINING <<<")

    # 🔹 SALVA SETTINGS
    with open("settings.json", "wb") as sf:
        linker.write_settings(sf)
    print("[dedupe_labels] settings.json salvato")
    print("[dedupe_labels] ===== FINE =====\n")

    return linker
