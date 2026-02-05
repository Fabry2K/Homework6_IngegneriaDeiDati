import dataset_utils as ds_utils
import create_ground_truth as gt
import mark_invalids as invalids
import utils
import create_gtv2 as gtv2

# ================================
# STEP 1: STATISTICHE SUI DATASET
# ================================

# ds_utils.count_nulls_and_uniques_big_csv("vehicles.csv", "vehicles")
# ds_utils.count_nulls_and_uniques_big_csv("used_cars_data.csv", "used_cars_data")

# ================================
# STEP 3: ALLINEAMENTO DEI DATASET
# ================================

# ds_utils.align_dataset_in_chunks("vehicles.csv", "vehicles_aligned.csv", dataset_type="vehicles")
# ds_utils.align_dataset_in_chunks("used_cars_data.csv", "used_cars_aligned.csv", dataset_type="used_cars")

# ds_utils.count_nulls_and_uniques_big_csv("vehicles_aligned.csv", "vehicles_aligned")
# ds_utils.count_nulls_and_uniques_big_csv("used_cars_aligned.csv", "used_cars_aligned")


# ======================================
# STEP 4.a: CONTROLLI SUI VIN
# ======================================

# invalids.mark_invalids("vehicles_aligned.csv", "vehicles_marked.csv")
# invalids.mark_invalids("used_cars_aligned.csv", "used_cars_marked.csv")


# utils.check_representativity("vehicles_marked.csv")
# utils.check_representativity("used_cars_marked.csv")

# ======================================
# STEP 4.a: CREAZIONE DELLA GROUND TRUTH
# ======================================

# gt.create_ground_truth(
#     vehicles_csv="vehicles_marked.csv",
#     used_cars_csv="used_cars_marked.csv",
#     output_csv="ground_truth.csv",
#     chunksize=100_000
# )

# gtv2.create_ground_truth()


# ===============================
# STEP 4b – RIMOZIONE VIN
# ===============================

utils.remove_vin(
    vehicles_input="vehicles_aligned.csv",
    used_cars_input="used_cars_aligned.csv",
    vehicles_output="vehicles_aligned.csv",
    used_cars_output="vehicles_aligned.csv"
)

utils.remove_vin_from_ground_truth(
    ground_truth_input="ground_truth.csv",
    ground_truth_output="ground_truth.csv"
)


# ===============================
# STEP 4c – SPLIT GROUND TRUTH
# ===============================

utils.split_ground_truth(
    ground_truth_input="ground_truth.csv",
    train_output="train.csv",
    val_output="validation.csv",
    test_output="test.csv"
)
