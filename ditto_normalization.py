import pandas as pd

FIELDS = [
    "manufacturer",
    "model",
    "year",
    "mileage",
    "fuel_type",
    "transmission",
    "body_type",
    "cylinders",
    "drive",
    "color"
]


def serialize_side(row, prefix):
    """
    Serializza un lato (a_ o b_) nel formato:
    COL field VAL value
    """
    tokens = []

    for field in FIELDS:
        col_name = f"{prefix}_{field}"
        val = row.get(col_name)

        if pd.isna(val):
            val_str = ""
        else:
            val_str = str(val).strip()
            val_str = val_str.replace("\t", " ").replace("\n", " ")

        tokens.append(f"COL {field} VAL {val_str}")

    return " ".join(tokens)


def csv_to_ditto_format(csv_path, output_txt_path):
    """
    Converte CSV con colonne:
    a_*, b_*, match
    nel formato richiesto dalla TUA versione di Ditto:

    left_record \t right_record \t label
    """

    # 🔥 Leggiamo tutto come stringa
    df = pd.read_csv(csv_path, dtype=str)

    with open(output_txt_path, "w", encoding="utf-8") as f_out:

        for _, row in df.iterrows():

            # label (0 o 1)
            label = str(row["match"]).strip()

            # serializzazione left e right
            left_str = serialize_side(row, "a")
            right_str = serialize_side(row, "b")

            # 🔥 ORDINE CORRETTO PER LA TUA VERSIONE DI DITTO
            line = f"{left_str}\t{right_str}\t{label}\n"

            f_out.write(line)

    print(f"File Ditto creato correttamente: {output_txt_path}")


def safe_str(val):
    if pd.isna(val) or str(val).lower() == "nan":
        return ""
    return str(val).strip()


def generate_ditto_input(candidates_csv, test_csv, output_txt):
    # Leggi i CSV come stringhe
    candidates = pd.read_csv(candidates_csv, dtype=str)
    test = pd.read_csv(test_csv, dtype=str)

    # Rimuovi colonne invalid
    candidates = candidates.drop(columns=['invalid_a', 'invalid_b'])

    # Aggiungi lato B colonne mancanti 'b_manufacturer' e 'b_year' copiando dal lato A
    candidates['b_manufacturer'] = candidates['manufacturer']
    candidates['b_year'] = candidates['year']

    # Rinominazioni per lato A e B (per merge e per Ditto)
    candidates = candidates.rename(columns={
        'manufacturer': 'a_manufacturer',
        'year': 'a_year',
        'model_a': 'a_model',
        'mileage_a': 'a_mileage',
        'fuel_type_a': 'a_fuel_type',
        'transmission_a': 'a_transmission',
        'body_type_a': 'a_body_type',
        'cylinders_a': 'a_cylinders',
        'drive_a': 'a_drive',
        'color_a': 'a_color',
        'model_b': 'b_model',
        'mileage_b': 'b_mileage',
        'fuel_type_b': 'b_fuel_type',
        'transmission_b': 'b_transmission',
        'body_type_b': 'b_body_type',
        'cylinders_b': 'b_cylinders',
        'drive_b': 'b_drive',
        'color_b': 'b_color'
    })

    # Merge con test set (mantiene tutte le righe del test set)
    merged = pd.merge(
        candidates,
        test,
        how='right',
        on=[
            'a_manufacturer', 'a_model', 'a_year', 'a_mileage', 'a_fuel_type',
            'a_transmission', 'a_body_type', 'a_cylinders', 'a_drive', 'a_color',
            'b_manufacturer', 'b_model', 'b_year', 'b_mileage', 'b_fuel_type',
            'b_transmission', 'b_body_type', 'b_cylinders', 'b_drive', 'b_color'
        ]
    )

    # Se la riga non era nei candidates, assegna label 0
    merged['match'] = merged['match'].fillna('0')

    # Funzione per convertire una riga in formato Ditto
    def row_to_ditto(row):
        s1 = (
            f"COL manufacturer VAL {row['a_manufacturer']} "
            f"COL model VAL {row['a_model']} "
            f"COL year VAL {row['a_year']} "
            f"COL mileage VAL {row['a_mileage']} "
            f"COL fuel_type VAL {row['a_fuel_type']} "
            f"COL transmission VAL {row['a_transmission']} "
            f"COL body_type VAL {row['a_body_type']} "
            f"COL cylinders VAL {row['a_cylinders']} "
            f"COL drive VAL {row['a_drive']} "
            f"COL color VAL {row['a_color']}"
        )
        s2 = (
            f"COL manufacturer VAL {row['b_manufacturer']} "
            f"COL model VAL {row['b_model']} "
            f"COL year VAL {row['b_year']} "
            f"COL mileage VAL {row['b_mileage']} "
            f"COL fuel_type VAL {row['b_fuel_type']} "
            f"COL transmission VAL {row['b_transmission']} "
            f"COL body_type VAL {row['b_body_type']} "
            f"COL cylinders VAL {row['b_cylinders']} "
            f"COL drive VAL {row['b_drive']} "
            f"COL color VAL {row['b_color']}"
        )
        label = row['match']
        return f"{s1}\t{s2}\t{label}\n"

    # Scrivi il file finale
    with open(output_txt, 'w', encoding='utf-8') as f:
        for _, row in merged.iterrows():
            f.write(row_to_ditto(row))

    print(f"File Ditto generato in: {output_txt}, righe: {len(merged)}")

