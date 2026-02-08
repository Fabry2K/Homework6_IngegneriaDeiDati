import pandas as pd
import re

# Dizionario per decimo carattere del VIN in base all'anno
year_to_code = {
    1980: 'A', 1981: 'B', 1982: 'C', 1983: 'D', 1984: 'E', 1985: 'F',
    1986: 'G', 1987: 'H', 1988: 'J', 1989: 'K', 1990: 'L', 1991: 'M',
    1992: 'N', 1993: 'P', 1994: 'R', 1995: 'S', 1996: 'T', 1997: 'V',
    1998: 'W', 1999: 'X', 2000: 'Y', 2001: '1', 2002: '2', 2003: '3',
    2004: '4', 2005: '5', 2006: '6', 2007: '7', 2008: '8', 2009: '9',
    2010: 'A', 2011: 'B', 2012: 'C', 2013: 'D', 2014: 'E', 2015: 'F',
    2016: 'G', 2017: 'H', 2018: 'J', 2019: 'K', 2020: 'L', 2021: 'M',
    2022: 'N', 2023: 'P', 2024: 'R', 2025: 'S', 2026: 'T', 2027: 'V',
    2028: 'W', 2029: 'X', 2030: 'Y', 2031: '1', 2032: '2', 2033: '3',
    2034: '4', 2035: '5', 2036: '6', 2037: '7', 2038: '8', 2039: '9'
}

# Regex per VIN valido (solo caratteri consentiti)
VIN_REGEX = re.compile(r'^[A-HJ-NPR-Z0-9]{17}$', re.IGNORECASE)

def mark_invalid_vin(input_csv, output_csv, chunksize=200_000):
    """
    Marca come invalidi i record sulla base di:
      1. VIN non lungo esattamente 17 caratteri
      2. VIN con caratteri non validi, o solo numerici o solo alfabetici
      3. Decimo carattere del VIN non coerente con l'anno
    Scrive il CSV aggiornato su output_csv.
    """
    print(f"Controllo VIN in corso su {input_csv}...")

    first_chunk = True

    # contatori per riepilogo
    total_valid = 0
    total_len_error = 0
    total_char_error = 0
    total_year_error = 0

    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        if 'invalid' not in chunk.columns:
            chunk['invalid'] = 0

        # assicuriamoci di trattare NA
        chunk['vin'] = chunk['vin'].fillna("").astype(str)

        # --- lunghezza VIN ---
        len_invalid_mask = chunk['vin'].str.len() != 17
        total_len_error += len_invalid_mask.sum()

        # --- caratteri validi ---
        char_invalid_mask = ~chunk['vin'].str.fullmatch(VIN_REGEX) | \
                            chunk['vin'].str.fullmatch(r'\d{17}') | \
                            chunk['vin'].str.fullmatch(r'[A-Za-z]{17}')
        total_char_error += char_invalid_mask.sum()

        # --- decimo carattere ---
        def check_decimo(row):
            vin = row['vin']
            year = row['year']
            # VIN troppo corto o anno mancante: non consideriamo qui
            if len(vin) < 10 or pd.isna(year):
                return False
            year = int(year)
            if year not in year_to_code:
                return False
            return vin[9] != year_to_code[year]

        year_invalid_mask = chunk.apply(check_decimo, axis=1)
        total_year_error += year_invalid_mask.sum()

        # --- unisci maschere ---
        invalid_mask = len_invalid_mask | char_invalid_mask | year_invalid_mask
        chunk['invalid'] = 0
        chunk.loc[invalid_mask, 'invalid'] = 1
        total_valid += (chunk['invalid'] == 0).sum()

        # scrivo chunk su CSV
        chunk.to_csv(
            output_csv,
            mode='w' if first_chunk else 'a',
            index=False,
            header=first_chunk
        )
        first_chunk = False
        print(f"Processato chunk di {len(chunk)} righe")

    # riepilogo finale
    print("\nRiepilogo record invalidi:")
    print(f"Record validi (invalid=0): {total_valid}")
    print(f"Record invalidi per lunghezza !=17: {total_len_error}")
    print(f"Record invalidi per caratteri VIN non validi: {total_char_error}")
    print(f"Record invalidi per decimo carattere errato: {total_year_error}")



def mark_invalid_duplicate_vins(input_csv, output_csv):
    """
    Aggiorna il campo 'invalid' mettendo 1 per tutti i record il cui VIN
    compare in più di un record. Rimangono con invalid=0 solo i VIN unici.
    Salva il CSV aggiornato su output_csv e stampa quanti record sono stati modificati.
    """
    # Carica tutto in memoria
    df = pd.read_csv(input_csv)

    if 'vin' not in df.columns:
        raise ValueError("'vin' column not found in CSV")
    if 'invalid' not in df.columns:
        df['invalid'] = 0  # inizializza la colonna se non presente

    # Conta occorrenze VIN
    vin_counts = df['vin'].value_counts()

    # Trova VIN duplicati
    duplicated_vins = vin_counts[vin_counts > 1].index

    # Maschera dei record da aggiornare
    mask = df['vin'].isin(duplicated_vins)

    # Conta quanti record verranno modificati
    num_modified = mask.sum()

    # Aggiorna il campo invalid
    df.loc[mask, 'invalid'] = 1

    # Salva il CSV aggiornato
    df.to_csv(output_csv, index=False)

    print(f"Record modificati (VIN duplicati marcati come invalid=1): {num_modified}")
    print(f"Record con VIN unico rimasti invalid=0: {(df['invalid'] == 0).sum()}")

    return df, num_modified
