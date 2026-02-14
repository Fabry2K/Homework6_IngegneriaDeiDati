import csv
import dedupe
import itertools

def to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def read_pairwise_dataset(filename):
    """
    Legge un CSV pairwise e crea due dataset separati (data_1, data_2)
    mantenendo le stesse regole di read_dataset:
    - numerici in float
    - stringhe strip + lower
    - valori vuoti in None
    """
    numeric_fields = {"year", "mileage", "price"}

    data_1 = {}
    data_2 = {}

    print(f"[read_pairwise_dataset] Caricamento CSV: {filename}")
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            record_a = {}
            record_b = {}

            for k, v in row.items():
                # conversione valore
                if v is None or v.strip() == "":
                    value = None
                elif k in numeric_fields:
                    value = to_float(v)
                else:
                    value = v.strip().lower()

                # campi condivisi tra A e B
                if k in ("manufacturer", "year"):
                    record_a[k] = value
                    record_b[k] = value
                # campi lato A (_a)
                elif k.endswith("_a"):
                    record_a[k[:-2]] = value  # rimuove _a
                # campi lato B (_b)
                elif k.endswith("_b"):
                    record_b[k[:-2]] = value  # rimuove _b
                # campi generici senza suffisso
                else:
                    record_a[k] = value
                    record_b[k] = value

            data_1[f"A_{i}"] = record_a
            data_2[f"B_{i}"] = record_b

    print(f"[read_pairwise_dataset] Record lato A: {len(data_1)}, lato B: {len(data_2)}")
    return data_1, data_2

def index_B1_pairwise(data_1, data_2, chunk_size=100000):
    """
    Assegna un indice univoco a ogni coppia (manufacturer, year)
    per due dataset distinti (record linkage).
    """
    print("[index_B1] Avvio blocking su (manufacturer, year)")
    pair_to_index = {}
    current_index = 0

    combined = list(data_1.items()) + list(data_2.items())
    total = len(combined)
    print(f"[index_B1] Totale record combinati: {total}")

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = combined[start:end]

        print(f"[index_B1]  - Processing records {start} → {end - 1}")

        for record_id, record in chunk:
            manufacturer = record.get("manufacturer")
            year = record.get("year")

            if manufacturer is None or year is None:
                record["block_index"] = None
                continue

            key = (manufacturer, year)
            if key not in pair_to_index:
                pair_to_index[key] = current_index
                current_index += 1

            record["block_index"] = pair_to_index[key]

        print(f"[index_B1]  - Chunk completato. Blocchi unici finora: {len(pair_to_index)}")

    print("[index_B1] Blocking completato")
    print(f"[index_B1] Totale blocchi unici creati: {len(pair_to_index)}")
    return data_1, data_2, pair_to_index

def read_groundtruth_pairwise(csv_file):
    """
    Legge un CSV groundtruth pairwise senza ID e restituisce un set di coppie
    frozenset([A_i, B_i]) con ID coerenti con read_pairwise_dataset.
    """
    groundtruth = set()
    print(f"[read_groundtruth_pairwise] Caricamento CSV ground truth: {csv_file}")
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, _ in enumerate(reader):
            a_id = f"A_{i}"
            b_id = f"B_{i}"
            groundtruth.add(frozenset([a_id, b_id]))
            if i < 5:  # stampa primi 5 esempi
                print(f"[groundtruth] riga {i}: {a_id}, {b_id}")

    print(f"[read_groundtruth_pairwise] Coppie ground truth totali: {len(groundtruth)}")
    return groundtruth



def evaluate_dedupe(filename_pairwise, settings_file, groundtruth_csv):
    """
    Valuta le prestazioni di dedupe RecordLink su un dataset pairwise
    confrontando i risultati con la groundtruth CSV.
    """
    print("[evaluate_dedupe] STEP 1: Carico dati pairwise")
    data_1, data_2 = read_pairwise_dataset(filename_pairwise)

    print("[evaluate_dedupe] STEP 2: Applico blocking")
    data_1, data_2, pair_to_index = index_B1_pairwise(data_1, data_2)

    print("[evaluate_dedupe] STEP 3: Carico modello RecordLink addestrato")
    with open(settings_file, "rb") as sf:
        linker = dedupe.StaticRecordLink(sf)

    print("[evaluate_dedupe] STEP 4: Calcolo soglia ottimale")
    threshold = linker.threshold(data_1, data_2, recall_weight=2.0)
    print(f"[evaluate_dedupe] Threshold calcolata: {threshold:.4f}")

    print("[evaluate_dedupe] STEP 5: Eseguo il record linkage")
    linked_records = linker.join(data_1, data_2, threshold)

    # Costruisco set di coppie predette
    predicted_pairs = set()
    for cluster, score in linked_records:
        if len(cluster) == 2:
            predicted_pairs.add(frozenset(cluster))
        else:
            for pair in itertools.combinations(cluster, 2):
                predicted_pairs.add(frozenset(pair))

    print(f"[evaluate_dedupe] Coppie predette totali: {len(predicted_pairs)}")
    if len(predicted_pairs) > 5:
        print(f"[evaluate_dedupe] Esempio prime 5 coppie predette: {list(predicted_pairs)[:5]}")

    # Ground truth
    groundtruth_pairs = read_groundtruth_pairwise(groundtruth_csv)

    # 🔹 Statistiche per blocco
    block_stats = {}
    for record_id, record in {**data_1, **data_2}.items():
        blk = record.get("block_index")
        if blk not in block_stats:
            block_stats[blk] = {"pred": 0, "gt": 0, "tp": 0, "example": (record.get("manufacturer"), record.get("year"))}

    # Conta predette e ground truth per blocco
    for pair in predicted_pairs:
        first = list(pair)[0]
        blk = data_1.get(first, data_2.get(first)).get("block_index")
        if blk is not None:
            block_stats[blk]["pred"] += 1
            if pair in groundtruth_pairs:
                block_stats[blk]["tp"] += 1

    for pair in groundtruth_pairs:
        first = list(pair)[0]
        blk = data_1.get(first, data_2.get(first)).get("block_index")
        if blk is not None:
            block_stats[blk]["gt"] += 1

    # Stampa dettagli blocco
    print("[evaluate_dedupe] Statistiche per blocco (manufacturer, year):")
    for blk, stats in sorted(block_stats.items()):
        man, year = stats["example"]
        print(f"  Block {blk} ({man}, {year}): predette={stats['pred']}, groundtruth={stats['gt']}, TP={stats['tp']}")

    # Metriche globali
    tp = len(predicted_pairs & groundtruth_pairs)
    fp = len(predicted_pairs - groundtruth_pairs)
    fn = len(groundtruth_pairs - predicted_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"[evaluate_dedupe] Risultati globali:")
    print(f"  True Positive: {tp}")
    print(f"  False Positive: {fp}")
    print(f"  False Negative: {fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    return precision, recall, f1
