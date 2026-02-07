import mediated_schema as ms
import create_ground_truth as gt
import mark_invalids as invalids
import utils
import create_gtv2 as gtv2



# OBIETTIVO: unificare gli annunci da diversi siti

# ================================
# STEP 1: STATISTICHE SUI DATASET
# ================================

# utils.count_nulls_and_uniques("vehicles.csv", "vehicles")
# utils.count_nulls_and_uniques("used_cars_data.csv", "used_cars_data")

# ================================
# STEP 3: ALLINEAMENTO DEI DATASET
# ================================

# 3.1: gestire VIN MANCANTI, al massimo si può controllare se si può ricavare dal secondo dataset

# 3.2: CLUSTERING per i duplicati. I Placeholder formano un cluster a sè, ma vengono comunque ignorati per la ground truth
utils.find_duplicates("vehicles.csv")


# 3.3: schema mediato: capire quali attributi servono (da entrambi i dataset), mantenere anche quelli degli annunci

# ms.align_dataset("vehicles.csv", "vehicles_aligned.csv", dataset_type="vehicles")
# ms.align_dataset("used_cars_data.csv", "used_cars_aligned.csv", dataset_type="used_cars")

# utils.count_nulls_and_uniques("vehicles.csv", "vehicles_aligned")
# utils.count_nulls_and_uniques("used_cars_data.csv", "used_cars_aligned")

# utils.check_representativity("vehicles_marked.csv")
# utils.check_representativity("used_cars_marked.csv")

# ======================================
# STEP 4.a: CREAZIONE DELLA GROUND TRUTH
# ======================================
#
# per ogni match si creano 5 non match
# la GROUND TRUTH va fatta confrontando i cluster ricavati dal primo file con i record del secondo file: 
# ogni cluster viene visto come una sola entità

# gt.create_ground_truth(
#     vehicles_csv="vehicles_marked.csv",
#     used_cars_csv="used_cars_marked.csv",
#     output_csv="ground_truth.csv",
#     chunksize=100_000
# )


# ===============================
# STEP 4b – RIMOZIONE VIN
# ===============================
#
# si rimuove il campo VIN dai file allineati e dalla GROUND TRUTH

# utils.remove_vin(
#     vehicles_input="vehicles_aligned.csv",
#     used_cars_input="used_cars_aligned.csv",
#     vehicles_output="vehicles_aligned.csv",
#     used_cars_output="vehicles_aligned.csv"
# )

# utils.remove_vin_from_ground_truth(
#     ground_truth_input="ground_truth.csv",
#     ground_truth_output="ground_truth.csv"
# )


# ===============================
# STEP 4c – SPLIT GROUND TRUTH
# ===============================
#
# split della GROUND TRUTH: 70 - 15 - 15 

# utils.split_ground_truth(
#     ground_truth_input="ground_truth.csv",
#     train_output="train.csv",
#     val_output="validation.csv",
#     test_output="test.csv"
# )
