import pandas as pd

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

def mark_invalids(input_csv, output_csv, chunksize=200_000):
    """
    Aggiorna il campo 'invalid' nel CSV passato in input in base a:
      1. VIN non lungo 17 caratteri
      2. VIN duplicati
      3. Decimo carattere del VIN non coerente con l'anno
    Salva il CSV aggiornato su output_csv.
    Stampa un riepilogo dei motivi dei record invalidi.
    """
    print(f"Controllo VIN in corso sul file {input_csv}...")

    # ======================
    # Passo 1: conteggio VIN duplicati globali
    # ======================
    vin_counts = {}
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, usecols=["vin"]):
        for vin in chunk["vin"]:
            if pd.isna(vin) or str(vin).strip() == "":
                continue
            vin_counts[vin] = vin_counts.get(vin, 0) + 1

    print(f"Totale VIN unici trovati: {len(vin_counts)}")

    # ======================
    # Passo 2: aggiorno 'invalid' per ciascun record
    # ======================
    first_chunk = True

    # contatori per riepilogo
    total_0 = 0
    total_1_missing = 0
    total_1_duplicate = 0
    total_1_year = 0

    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        if 'invalid' not in chunk.columns:
            chunk['invalid'] = 0

        for idx, row in chunk.iterrows():
            vin = str(row["vin"]).strip() if pd.notna(row["vin"]) else ""
            year = row["year"] if "year" in row else None

            invalid = 0

            # 1️⃣ VIN assente o lunghezza diversa da 17
            if len(vin) != 17:
                invalid = 1
                total_1_missing += 1

            # 2️⃣ VIN duplicato
            elif vin_counts.get(vin, 0) > 1:
                invalid = 1
                total_1_duplicate += 1

            # 3️⃣ Decimo carattere VIN non coerente con anno
            elif pd.notna(year) and int(year) in year_to_code:
                expected_char = year_to_code[int(year)]
                if vin[9] != expected_char:  # decimo carattere
                    invalid = 1
                    total_1_year += 1

            if invalid == 0:
                total_0 += 1

            chunk.at[idx, 'invalid'] = invalid

        # Scrivo chunk su CSV
        chunk.to_csv(
            output_csv,
            mode='w' if first_chunk else 'a',
            index=False,
            header=first_chunk
        )
        first_chunk = False
        print(f"Processato chunk di {len(chunk)} righe")

    # ======================
    # Riepilogo finale
    # ======================
    print("\nRiepilogo controlli VIN:")
    print(f"Record validi (invalid=0): {total_0}")
    print(f"Record invalidi per VIN assente/lunghezza !=17: {total_1_missing}")
    print(f"Record invalidi per VIN duplicato: {total_1_duplicate}")
    print(f"Record invalidi per decimo carattere VIN errato: {total_1_year}")


