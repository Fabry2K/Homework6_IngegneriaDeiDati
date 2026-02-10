import dedupe

def couples_to_python_dictionary(ground_truth):

    labeled_pairs = []

    for _,row in ground_truth.iterrows():
        #dataset A
        dataset_a = {col.replace('a_',''): row[col] for col in ground_truth.columns if col.startswith('a_')}
        #dataset B
        dataset_b = {col.replace('b_',''): row[col] for col in ground_truth.columns if col.startswith('b_')}

        # match = True/False
        match = bool(row['match'])

        # aggiungiamo la coppia etichettata
        labeled_pairs.append((dataset_a, dataset_b, match))