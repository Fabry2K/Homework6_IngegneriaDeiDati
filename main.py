import mediated_schema as ms
import ground_truth as gt
import vin_checks as checks
import normalization as norm
import utils
import pandas as pd



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

# 3.3: set del campo invalid = 1 per record con: VIN 

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

# dataset = pd.read_csv('vehicles_final.csv')
# print(dataset['model'].value_counts())

# dataset = pd.read_csv('used_cars_aligned.csv')
# print(dataset['model'].value_counts())



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
#    chunksize=200_000,
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
# strategia B1: su manufacturer e anno
# strategia B2: ___




