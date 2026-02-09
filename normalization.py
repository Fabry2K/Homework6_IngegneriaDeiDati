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
    convert_all_to_string(input_csv, output_csv)

def convert_all_to_string(input_csv, output_csv):
    """
    Converte tutti i campi di un CSV in stringhe.
    - Se il valore è un numero con decimale tipo 4.0, diventa '4'
    - Tutti i valori finali sono stringhe
    """
    print(f"Caricamento file {input_csv}...")
    df = pd.read_csv(input_csv, dtype=str)  # inizialmente tutto come string per evitare warning
    
    def to_str_clean(x):
        if pd.isna(x):
            return ""
        try:
            val = float(x)
            val_int = int(val)
            return str(val_int)
        except:
            return str(x).strip()
    
    print("Conversione di tutti i campi in stringa...")
    for col in df.columns:
        df[col] = df[col].apply(to_str_clean)
    
    print(f"Salvataggio file normalizzato in {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Conversione completata.")

# Funzione generica per convertire in stringa rimuovendo decimali se è un numero
def to_string_clean(x):
    if pd.isna(x):
        return ""
    try:
        val = float(x)
        val_int = int(val)
        return str(val_int)
    except:
        return str(x).strip()


# Normalizza manufacturer
def normalize_manufacturer(input_csv, output_csv=None):
    print(f"Normalizzazione manufacturer su {input_csv}...")
    df = pd.read_csv(input_csv, dtype=str)
    if 'manufacturer' not in df.columns:
        raise ValueError("Il CSV non contiene la colonna 'manufacturer'")
    df['manufacturer'] = df['manufacturer'].astype(str).str.lower().str.replace('-', '', regex=False).str.replace(' ', '', regex=False)
    # Tutto string
    df = df.astype(str)
    out_path = output_csv if output_csv else input_csv
    df.to_csv(out_path, index=False)
    print(f"File salvato: {out_path}")


# Normalizza fuel_type
def normalize_fuel_type_csv(input_csv, output_csv, fuel_col='fuel_type'):
    print(f"Normalizzazione fuel_type su {input_csv}...")
    df = pd.read_csv(input_csv, dtype=str)
    def normalize(value):
        val = str(value).strip().lower()
        if val == 'gas':
            return 'gasoline'
        return val
    df[fuel_col] = df[fuel_col].apply(normalize)
    df = df.astype(str)
    df.to_csv(output_csv, index=False)
    print(f"File salvato: {output_csv}")


# Normalizza model
def normalize_model_csv(input_csv, output_csv, model_col='model'):
    print(f"Normalizzazione model su {input_csv}...")
    df = pd.read_csv(input_csv, dtype=str)
    df[model_col] = df[model_col].astype(str).str.lower()
    df = df.astype(str)
    df.to_csv(output_csv, index=False)
    print(f"File salvato: {output_csv}")


# Normalizza transmission
def normalize_transmission_csv(input_csv, output_csv, transmission_col='transmission'):
    print(f"Normalizzazione transmission su {input_csv}...")
    df = pd.read_csv(input_csv, dtype=str)
    df[transmission_col] = df[transmission_col].astype(str).str.lower()
    mapping = {'a': 'automatic', 'm': 'manual', 'cvt': 'other', 'dual clutch': 'other'}
    df[transmission_col] = df[transmission_col].apply(lambda x: mapping.get(x, x))
    df = df.astype(str)
    df.to_csv(output_csv, index=False)
    print(f"File salvato: {output_csv}")


# Normalizza body_type
def normalize_body_type_csv(input_csv, output_csv, body_type_col='body_type'):
    print(f"Normalizzazione body_type su {input_csv}...")
    df = pd.read_csv(input_csv, dtype=str)
    df[body_type_col] = df[body_type_col].astype(str).str.lower()
    df[body_type_col] = df[body_type_col].replace({'suv / crossover': 'suv', 'pickup truck': 'pickup'})
    df[body_type_col] = df[body_type_col].str.replace(' ', '', regex=True).str.replace('-', '', regex=True)
    df = df.astype(str)
    df.to_csv(output_csv, index=False)
    print(f"File salvato: {output_csv}")


# Normalizza cylinders
def normalize_cylinders_csv(input_csv, output_csv, cylinders_col='cylinders'):
    print(f"Normalizzazione cylinders su {input_csv}...")
    df = pd.read_csv(input_csv, dtype=str)
    def extract_cylinders(value):
        if pd.isna(value):
            return ""
        s = str(value).lower()
        match = re.search(r'\d+', s)
        if match:
            num = match.group()
            if num == '2':
                return 'other'
            return str(int(float(num)))
        else:
            return 'other'
    df[cylinders_col] = df[cylinders_col].apply(extract_cylinders)
    df = df.astype(str)
    df.to_csv(output_csv, index=False)
    print(f"File salvato: {output_csv}")


# Normalizza drive
def normalize_drive_csv(input_csv, output_csv, drive_col='drive'):
    print(f"Normalizzazione drive su {input_csv}...")
    df = pd.read_csv(input_csv, dtype=str)
    df[drive_col] = df[drive_col].astype(str).str.lower()
    df = df.astype(str)
    df.to_csv(output_csv, index=False)
    print(f"File salvato: {output_csv}")


# Normalizza color
def normalize_color_csv(input_csv, output_csv, color_col='color'):
    print(f"Normalizzazione color su {input_csv}...")
    standard_colors = ["white","black","silver","blue","grey","red","green","custom","brown","yellow","orange","purple"]
    synonyms = {"gray": "grey"}
    df = pd.read_csv(input_csv, dtype=str)
    def normalize(value):
        val_lower = str(value).lower()
        for syn, std in synonyms.items():
            val_lower = val_lower.replace(syn, std)
        for color in standard_colors:
            if color in val_lower:
                return color
        return val_lower
    df[color_col] = df[color_col].apply(normalize)
    df = df.astype(str)
    df.to_csv(output_csv, index=False)
    print(f"File salvato: {output_csv}")


# Normalizza year (tutto stringa senza decimale)
def normalize_year_csv(input_csv, output_csv, year_col='year'):
    print(f"Normalizzazione year su {input_csv}...")
    df = pd.read_csv(input_csv, dtype=str)
    df[year_col] = df[year_col].apply(to_string_clean)
    df = df.astype(str)
    df.to_csv(output_csv, index=False)
    print(f"File salvato: {output_csv}")
