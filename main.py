import dataset_stats as stats
import align_dataset 
import create_ground_truth as gt
import mark_invalids as invalids
import check_representativity as representativity

# ================================
# STEP 1: STATISTICHE SUI DATASET
# ================================

# stats.count_nulls_and_uniques_big_csv("vehicles.csv", "vehicles")
# stats.count_nulls_and_uniques_big_csv("used_cars_data.csv", "used_cars_data")

# ================================
# STEP 3: ALLINEAMENTO DEI DATASET
# ================================

# align_dataset.align_dataset_in_chunks("vehicles.csv", "vehicles_aligned.csv", dataset_type="vehicles")
# align_dataset.align_dataset_in_chunks("used_cars_data.csv", "used_cars_aligned.csv", dataset_type="used_cars")

# stats.count_nulls_and_uniques_big_csv("vehicles_aligned.csv", "vehicles_aligned")
# stats.count_nulls_and_uniques_big_csv("used_cars_aligned.csv", "used_cars_aligned")


# ======================================
# STEP 4.a: CONTROLLI SUI VIN
# ======================================

# invalids.mark_invalids("vehicles_aligned.csv", "vehicles_marked.csv")
# invalids.mark_invalids("used_cars_aligned.csv", "used_cars_marked.csv")


# representativity.check_representativity("vehicles_marked.csv")
# representativity.check_representativity("used_cars_marked.csv")

# ======================================
# STEP 4.a: CREAZIONE DELLA GROUND TRUTH
# ======================================

gt.create_ground_truth(
    vehicles_csv="vehicles_marked.csv",
    used_cars_csv="used_cars_marked.csv",
    output_csv="ground_truth.csv",
    chunksize=100_000
)
