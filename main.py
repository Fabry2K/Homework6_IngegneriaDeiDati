import mediated_schema as ms
import ground_truth as gt
import vin_checks as checks
import normalization as norm
import utils
import pandas as pd
import blocking
import record_linkage as rl



# OBIETTIVO: unificare gli annunci da diversi siti

# ================================
# STEP 1: STATISTICHE SUI DATASET
# ================================

# utils.count_nulls_and_uniques("vehicles.csv", "vehicles")
# utils.count_nulls_and_uniques("used_cars_data.csv", "used_cars_data")

# ================================
# STEP 3: ALLINEAMENTO DEI DATASET
# ================================

# 3.1: CLUSTERING per i duplicati. I Placeholder formano un cluster a sè, ma vengono comunque ignorati per la ground truth
# utils.deduplicate_csv(
#    csv_path="vehicles.csv",
#    csv_out_clean="vehicles_cleaned.csv",
#    csv_out_duplicates="vehicles_ispezione.csv",
#    desc_threshold=0.7
# )

# utils.count_nulls_and_uniques("vehicles.csv", "vehicles")
# utils.count_nulls_and_uniques("vehicles_cleaned.csv", "vehicles_cleaned")


# 3.2: schema mediato: capire quali attributi servono (da entrambi i dataset), mantenere anche quelli degli annunci

# ms.align_dataset("vehicles_cleaned.csv", "vehicles_aligned.csv", dataset_type="vehicles")
# ms.align_dataset("used_cars_data.csv", "used_cars_aligned.csv", dataset_type="used_cars")


# utils.count_nulls_and_uniques("vehicles_aligned.csv", "vehicles_aligned")
# utils.count_nulls_and_uniques("used_cars_aligned.csv", "used_cars_aligned")

# 3.3: set del campo invalid = 1 per record 

# checks.mark_invalid_vin('vehicles_aligned.csv','vehicles_marked.csv',200000)
# checks.mark_invalid_duplicate_vins('vehicles_marked.csv','vehicles_final.csv')



# ===============================
# NORMALIZZAZIONE DEI CAMPI
# ===============================


# norm.normalize_all(
#     input_csv = 'vehicles_final.csv', 
#     output_csv = 'vehicles_final.csv'
# )

# norm.normalize_all(
#     input_csv = 'used_cars_aligned.csv', 
#     output_csv = 'used_cars_aligned.csv'
# )

# dataset = pd.read_csv('vehicles_final.csv', dtype=str)
# print(dataset['cylinders'].value_counts())

# dataset = pd.read_csv('used_cars_aligned.csv', dtype=str)
# print(dataset['cylinders'].value_counts())



# ======================================
# STEP 4.a: CREAZIONE DELLA GROUND TRUTH
# ======================================
#
# per ogni match si creano 2 non match
# per la creazione dei match e dei non match vengono considerati solo i record con Invalid = 0

# gt.build_ground_truth(
#    file_a = 'vehicles_final.csv',
#    file_b = 'used_cars_aligned.csv',
#    output_gt = 'ground_truth.csv',
#    chunksize=500_000,
#    negatives_per_match=2,
#    random_seed=42
# )




# ===============================
# STEP 4b – RIMOZIONE VIN
# ===============================
#
# si rimuove il campo VIN dai file allineati e dalla GROUND TRUTH

# utils.remove_vin_from_dataset(
#     "vehicles_final.csv",
#     "vehicles_final.csv"
# )

# utils.remove_vin_from_dataset(
#     "used_cars_aligned.csv",
#     "used_cars_final.csv"
# )

# utils.remove_vins_from_ground_truth(
#     "ground_truth.csv",
#     "ground_truth_final.csv"
# )

# Normalizzazione float
# norm.normalize_numeric_fields_no_decimal(
#     input_csv = 'vehicles_final.csv', 
#     output_csv = 'vehicles_final.csv'
# )

# norm.normalize_numeric_fields_no_decimal(
#     input_csv = 'used_cars_final.csv', 
#     output_csv = 'used_cars_final.csv'
# )

# norm.normalize_gt_numeric_fields_no_decimal(
#     input_gt_csv = 'ground_truth_final.csv', 
#     output_gt_csv = 'ground_truth_final.csv'
# )

# ===============================
# STEP 4c – SPLIT GROUND TRUTH
# ===============================
#
# split della GROUND TRUTH: 70 - 15 - 15 

# utils.split_ground_truth(
#    input_gt = 'ground_truth_final.csv',
#    train_out = 'train.csv',
#    val_out = 'validation.csv',
#    test_out = 'test.csv',
#    train_ratio=0.7,
#    val_ratio=0.2,
#    test_ratio=0.1,
#    seed=42
# )


# ===============================
# STEP 4d – STRATEGIE DI BLOCKING
# ===============================
#
# strategia B1: su manufacturer e year
# strategia B2: su transmission, fuel type e year 

# Calcolo candidate pairs con blocking B1
# blocking.generate_candidate_pairs_B1(
#     file_a="vehicles_final.csv",
#     file_b="used_cars_final.csv",
#     output_file="D:\HM6\candidate_pairs_B1.csv",
#     chunk_size=100_000
# )


# Calcolo candidate pairs con blocking B2
blocking.generate_candidate_pairs_B2(
    file_a="vehicles_final.csv",
    file_b="used_cars_final.csv",
    output_file="candidate_pairs_B2.csv",
    chunk_size=100_000
)


# ===================================
# STEP 4e – REGOLE PER RECORD LINKAGE
# ===================================
#
# manufacturer -> esattamente uguale
# model -> si calcola la somiglianza cosine similarity. più sono uguali le stringhe più punti prende. una similarità alta qui deve valere più di un punteggio pieno negli altri campi
# year -> esattamente uguale
# mileage -> punteggio in base alla distnza fra i due valori, più è diverso meno punti si conferisce
# fuel_type -> esattamente uguale, ma ("flex fuel vehicle" = "gasoline"), ("biodiesel" = "diesel"), ("other" match con tutto, ma pochi punti)
# transmission -> esattamente uguale
# body_type -> esattamente uguale, ma ("truck" = "pickup"), ("offroad" è uguale a  "suv" e "pickup"), ("other" match con tutto, ma pochi punti)
# cylinders -> esattamente uguale, ("other" fa match con tutto, ma meno punti)
# drive -> esattamente uguale (con 4wd = awd), (fwd/rwd corrispondono anche a 4x2)
# color -> se uguale punteggio pieno, altrimenti dai punteggio molto molto basso


# rl.evaluate_B1(
#     'D:\HM6\candidate_pairs_B1.csv', 
#     'test.csv', 
#     chunk_size=1000000, 
#     match_threshold=0.70
# )


# if __name__ == "__main__":

#     rl.evaluate_B1_parallel(
#         r'D:\HM6\candidate_pairs_B1.csv',  # usa raw string r'...' per evitare \H
#         r'test.csv',
#         chunk_size=500000,
#         match_threshold=0.70,
#         max_workers=8, 
#         backup_file='B1_recordLinkage.csv'
#     )



