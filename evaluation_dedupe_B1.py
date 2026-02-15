import csv
import dedupe
import itertools


def to_float(val):
    """Converte una stringa in float. Rimuove spazi e virgole. Restituisce None se non convertibile."""
    if val is None:
        return None
    try:
        val = val.replace(",", "").strip()
        if val == "":
            return None
        return float(val)
    except (ValueError, AttributeError):
        return None


def read_pairwise_dataset(filename, verbose_every=1000):
    """Legge un CSV pairwise e crea due dataset separati (data_1, data_2)."""
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
                # Conversione valore
                if v is None or v.strip() == "":
                    value = None
                elif k in numeric_fields:
                    value = to_float(v)
                    if value is None:
                        print(f"[DEBUG] riga {i}, campo {k} non convertibile: '{v}'")
                else:
                    value = v.strip().lower()

                # Campi condivisi tra A e B
                if k in ("manufacturer", "year"):
                    record_a[k] = value
                    record_b[k] = value
                elif k.endswith("_a"):
                    record_a[k[:-2]] = value
                elif k.endswith("_b"):
                    record_b[k[:-2]] = value
                else:
                    record_a[k] = value
                    record_b[k] = value

            data_1[f"A_{i}"] = record_a
            data_2[f"B_{i}"] = record_b

            if i < 5:
                print(f"[pairwise] riga {i}: A_{i}, B_{i}")
            elif i % verbose_every == 0:
                print(f"[pairwise] riga {i} letta...")

    print(f"[read_pairwise_dataset] Record lato A: {len(data_1)}, lato B: {len(data_2)}")
    return data_1, data_2


def clean_numeric_fields(data, numeric_fields={"year", "mileage", "price"}):
    """Assicura che tutti i campi numerici siano float o None."""
    for record_id, record in data.items():
        for field in numeric_fields:
            val = record.get(field)
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                record[field] = to_float(str(val))
    return data


def index_B1_pairwise(data_1, data_2, chunk_size=100000):
    """Blocking su (manufacturer, year) per due dataset distinti."""
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


def read_groundtruth_pairwise(csv_file, valid_ids=None, verbose_every=1000):
    """
    Legge CSV groundtruth pairwise e filtra ID non presenti nel dataset.
    valid_ids: set degli ID effettivamente presenti in data_1/data_2
    """
    groundtruth = set()
    print(f"[read_groundtruth_pairwise] Caricamento CSV ground truth: {csv_file}")
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, _ in enumerate(reader):
            a_id = f"A_{i}"
            b_id = f"B_{i}"

            # Ignora ID non validi
            if valid_ids is not None and (a_id not in valid_ids or b_id not in valid_ids):
                continue

            groundtruth.add(frozenset([a_id, b_id]))

            if i < 5:
                print(f"[groundtruth] riga {i}: {a_id}, {b_id}")
            elif i % verbose_every == 0:
                print(f"[groundtruth] riga {i} letta...")

    print(f"[read_groundtruth_pairwise] Coppie ground truth totali filtrate: {len(groundtruth)}")
    return groundtruth


def evaluate_dedupe(filename_pairwise, settings_file, groundtruth_csv, use_static=True, manual_threshold=0.9):
    """Valuta le prestazioni di dedupe RecordLink su dataset pairwise."""
    print("[evaluate_dedupe] STEP 1: Carico dati pairwise")
    data_1, data_2 = read_pairwise_dataset(filename_pairwise)

    # Pulizia campi numerici
    data_1 = clean_numeric_fields(data_1)
    data_2 = clean_numeric_fields(data_2)

    print("[evaluate_dedupe] STEP 2: Applico blocking")
    data_1, data_2, pair_to_index = index_B1_pairwise(data_1, data_2)

    print("[evaluate_dedupe] STEP 3: Carico modello RecordLink addestrato")
    with open(settings_file, "rb") as sf:
        linker = dedupe.StaticRecordLink(sf) if use_static else dedupe.RecordLink(sf)

    # STEP 4: soglia
    threshold = manual_threshold if use_static else linker.threshold(data_1, data_2, recall_weight=2.0)
    print(f"[evaluate_dedupe] Usando soglia: {threshold}")

    print("[evaluate_dedupe] STEP 5: Eseguo il record linkage")
    linked_records = linker.join(data_1, data_2, threshold)

    predicted_pairs = set()
    for cluster, score in linked_records:
        for pair in itertools.combinations(cluster, 2):
            predicted_pairs.add(frozenset(pair))

    print(f"[evaluate_dedupe] Coppie predette totali: {len(predicted_pairs)}")

    # IDs validi
    all_ids = set(data_1.keys()) | set(data_2.keys())
    groundtruth_pairs = read_groundtruth_pairwise(groundtruth_csv, valid_ids=all_ids)

    # Statistiche per blocco
    block_stats = {}
    for record_id, record in {**data_1, **data_2}.items():
        blk = record.get("block_index")
        if blk not in block_stats:
            block_stats[blk] = {
                "pred": 0,
                "gt": 0,
                "tp": 0,
                "example": (record.get("manufacturer"), record.get("year"))
            }

    for pair in predicted_pairs:
        first = list(pair)[0]
        record = data_1.get(first) or data_2.get(first)
        if record is None:
            continue
        blk = record.get("block_index")
        if blk is not None:
            block_stats[blk]["pred"] += 1
            if pair in groundtruth_pairs:
                block_stats[blk]["tp"] += 1

    for pair in groundtruth_pairs:
        first = list(pair)[0]
        record = data_1.get(first) or data_2.get(first)
        if record is None:
            continue
        blk = record.get("block_index")
        if blk is not None:
            block_stats[blk]["gt"] += 1

    # Stampa dettagli blocco
    print("[evaluate_dedupe] Statistiche per blocco:")
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
