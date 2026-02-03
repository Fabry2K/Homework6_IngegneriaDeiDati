import dataset_stats as stats
import align_dataset 

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


stats.count_nulls_and_uniques_big_csv("vehicles_aligned.csv", "vehicles_aligned")
stats.count_nulls_and_uniques_big_csv("used_cars_aligned.csv", "used_cars_aligned")