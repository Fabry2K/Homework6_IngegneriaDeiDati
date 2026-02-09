import pandas as pd
import re

def normalize_all(input_csv, output_csv=None):
    normalize_manufacturer(input_csv, output_csv)
    normalize_fuel_type_csv(input_csv, output_csv)
    normalize_color_csv(input_csv, output_csv)
    normalize_drive_csv(input_csv, output_csv)
    normalize_transmission_csv(input_csv, output_csv)
    normalize_body_type_csv(input_csv, output_csv)
    normalize_cylinders_csv(input_csv, output_csv)
    normalize_model_csv(input_csv, output_csv)


# normalize 'manufacturer'
def normalize_manufacturer(input_csv, output_csv=None):
    """
    Normalizza il campo 'manufacturer' in un CSV:
    - tutto in lowercase
    - rimuove trattini '-'
    - rimuove tutti gli spazi (iniziali, finali e in mezzo)
    
    Se output_csv non è specificato, sovrascrive il file di input.
    """
    print(f"Normalizzazione manufacturer in corso su {input_csv}...")
    
    # Legge il CSV
    df = pd.read_csv(input_csv)

    if 'manufacturer' not in df.columns:
        raise ValueError("Il CSV non contiene la colonna 'manufacturer'")

    # Normalizzazione
    df['manufacturer'] = (
        df['manufacturer']
        .astype(str)  # assicurarsi che sia stringa
        .str.lower()  # tutto lowercase
        .str.replace('-', '', regex=False)  # rimuove trattini
        .str.replace(' ', '', regex=False)  # rimuove tutti gli spazi
    )

    # Scrive il file
    out_path = output_csv if output_csv else input_csv
    df.to_csv(out_path, index=False)
    print(f"Normalizzazione completata. File salvato in: {out_path}")


# normalize 'fuel_type'
def normalize_fuel_type_csv(input_csv, output_csv, fuel_col='fuel_type'):
    """
    Normalizza il campo fuel_type di un CSV:
    - tutto lowercase
    - 'gas' diventa 'gasoline'
    - rimuove spazi inutili
    Salva il CSV aggiornato su output_csv.
    """
    print(f"Normalizzazione fuel_type in corso sul file {input_csv}...")
    
    df = pd.read_csv(input_csv)
    
    def normalize(value):
        if pd.isna(value):
            return value
        val = str(value).strip().lower()
        if val == 'gas':
            return 'gasoline'
        return val

    df[fuel_col] = df[fuel_col].apply(normalize)
    
    df.to_csv(output_csv, index=False)
    print(f"File normalizzato salvato in {output_csv}")


# normalize 'model'
def normalize_model_csv(input_csv, output_csv, model_col='model'):
    """
    Normalizza il campo 'model' portando tutto in lowercase.
    """
    # Leggi CSV
    df = pd.read_csv(input_csv)

    # Trasforma in lowercase
    df[model_col] = df[model_col].astype(str).str.lower()

    # Salva il CSV normalizzato
    df.to_csv(output_csv, index=False)
    print(f"File normalizzato salvato in {output_csv}")


# normalize 'transmission'
def normalize_transmission_csv(input_csv, output_csv, transmission_col='transmission'):
    """
    Normalizza il campo 'transmission' di un CSV:
    - 'A' → 'automatic'
    - 'M' → 'manual'
    - 'CVT', 'Dual Clutch' → 'other'
    - tutto lowercase
    """
    # Leggi CSV
    df = pd.read_csv(input_csv)

    # Trasforma tutto in lowercase
    df[transmission_col] = df[transmission_col].astype(str).str.lower()

    # Dizionario di mapping
    mapping = {
        'a': 'automatic',
        'm': 'manual',
        'cvt': 'other',
        'dual clutch': 'other'
    }

    # Applica la mappatura, se il valore non è nel dizionario lo lasciamo lowercase così com'è
    df[transmission_col] = df[transmission_col].apply(lambda x: mapping.get(x, x))

    # Salva il CSV normalizzato
    df.to_csv(output_csv, index=False)
    print(f"File normalizzato salvato in {output_csv}")


# normalize 'body_type'
def normalize_body_type_csv(input_csv, output_csv, body_type_col='body_type'):
    """
    Normalizza il campo 'body_type' di un CSV:
    - Tutto lowercase
    - Rimuove spazi bianchi (anche interni) e trattini
    - 'SUV / Crossover' diventa 'suv'
    - 'Pickup Truck' diventa 'pickup'
    """
    # Leggi CSV
    df = pd.read_csv(input_csv)

    # Trasforma tutto in lowercase
    df[body_type_col] = df[body_type_col].astype(str).str.lower()

    # Sostituzioni specifiche
    df[body_type_col] = df[body_type_col].replace({
        'suv / crossover': 'suv',
        'pickup truck': 'pickup'
    })

    # Rimuovi spazi bianchi e trattini residui
    df[body_type_col] = df[body_type_col].str.replace(' ', '', regex=True)
    df[body_type_col] = df[body_type_col].str.replace('-', '', regex=True)

    # Salva CSV normalizzato
    df.to_csv(output_csv, index=False)
    print(f"File normalizzato salvato in {output_csv}")


# normalize 'cylinders'
def normalize_cylinders_csv(input_csv, output_csv, cylinders_col='cylinders'):
    """
    Normalizza il campo 'cylinders' in un CSV:
    - Estrae solo il numero dai valori tipo 'V6', 'I4', '6 cylinders', ecc.
    - Restituisce il numero come stringa, senza decimali.
    - Sostituisce '2' con 'other'.
    - Mantiene gli altri numeri così come sono.
    """
    # Leggi CSV
    df = pd.read_csv(input_csv)

    def extract_cylinders(value):
        if pd.isna(value):
            return pd.NA
        s = str(value).lower()
        # Cerca un numero nella stringa
        match = re.search(r'\d+', s)
        if match:
            num = match.group()
            if num == '2':
                return 'other'
            return str(int(num))
        else:
            return 'other'

    # Applica la funzione
    df[cylinders_col] = df[cylinders_col].apply(extract_cylinders)

    # Salva CSV normalizzato
    df.to_csv(output_csv, index=False)
    print(f"File normalizzato salvato in {output_csv}")

# normalize 'drive'
def normalize_drive_csv(input_csv, output_csv, drive_col='drive'):
    """
    Normalizza il campo 'drive' di un CSV semplicemente portandolo tutto in lowercase.
    """
    # Leggi CSV
    df = pd.read_csv(input_csv)

    # Trasforma in lowercase
    df[drive_col] = df[drive_col].astype(str).str.lower()

    # Salva il CSV normalizzato
    df.to_csv(output_csv, index=False)
    print(f"File normalizzato salvato in {output_csv}")


# normalize 'color'
def normalize_color_csv(input_csv, output_csv, color_col='color'):
    """
    Normalizza il campo 'color' di un CSV usando una lista di colori standard.
    - Tutto lowercase
    - Cerca il colore anche all'interno di parole unite
    - Sostituisce 'gray' con 'grey'
    - Se non trova match, lascia la stringa in lowercase
    """
    # Lista dei colori standard
    standard_colors = [
        "white", "black", "silver", "blue", "grey",
        "red", "green", "custom", "brown", "yellow",
        "orange", "purple"
    ]
    
    # Mappa sinonimi
    synonyms = {
        "gray": "grey"
    }
    
    print(f"Normalizzazione colori in corso sul file {input_csv}...")
    
    df = pd.read_csv(input_csv)

    def normalize(value):
        if pd.isna(value):
            return value
        val_lower = str(value).lower()
        # gestisci sinonimi
        for syn, std in synonyms.items():
            val_lower = val_lower.replace(syn, std)
        # cerca colore standard
        for color in standard_colors:
            if color in val_lower:
                return color
        return val_lower  # se nessun colore trovato, mantieni lowercase

    df[color_col] = df[color_col].apply(normalize)
    
    df.to_csv(output_csv, index=False)
    print(f"File normalizzato salvato in {output_csv}")


def normalize_numeric_fields_no_decimal(
    input_csv,
    output_csv,
    cols=("mileage", "year", "cylinders")
):
    """
    Normalizza i campi numerici specificati:
    - rimuove '.0' e qualsiasi parte decimale
    - converte tutto in stringa
    - lascia invariati valori non numerici (es. 'other')
    """

    print(f"📥 Caricamento file {input_csv}...")
    df = pd.read_csv(input_csv)

    for col in cols:
        if col not in df.columns:
            print(f"⚠ Colonna '{col}' non trovata, salto.")
            continue

        print(f"🔧 Normalizzazione colonna '{col}'...")

        def normalize_value(x):
            if pd.isna(x):
                return x
            try:
                # converte in float → int → string
                return str(int(float(x)))
            except:
                # non numerico: lo restituisce come stringa pulita
                return str(x).strip()

        df[col] = df[col].apply(normalize_value)

    print(f"💾 Salvataggio file normalizzato in {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("✅ Normalizzazione completata.")


def normalize_gt_numeric_fields_no_decimal(
    input_gt_csv,
    output_gt_csv
):
    """
    Normalizza mileage, year e cylinders nella ground truth:
    - a_* e b_*
    - rimuove decimali (es. 2018.0 -> '2018')
    - mantiene valori non numerici (es. 'other')
    - tutto come stringa
    """

    print(f"📥 Caricamento ground truth: {input_gt_csv}")
    df = pd.read_csv(input_gt_csv)

    # colonne da normalizzare
    base_cols = ["year", "mileage", "cylinders"]
    cols_to_fix = []

    for base in base_cols:
        cols_to_fix.append(f"a_{base}")
        cols_to_fix.append(f"b_{base}")

    def normalize_value(x):
        if pd.isna(x):
            return x
        try:
            return str(int(float(x)))
        except:
            return str(x).strip()

    for col in cols_to_fix:
        if col not in df.columns:
            print(f"⚠ Colonna '{col}' non trovata, salto.")
            continue

        print(f"🔧 Normalizzazione colonna '{col}'...")
        df[col] = df[col].apply(normalize_value)

    print(f"💾 Salvataggio ground truth normalizzata in {output_gt_csv}")
    df.to_csv(output_gt_csv, index=False)
    print("✅ Normalizzazione ground truth completata.")
