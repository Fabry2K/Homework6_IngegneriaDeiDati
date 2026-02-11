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
