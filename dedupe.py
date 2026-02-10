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

    return labeled_pairs


def fields_to_dedupe_format():

    fields = [
        {'field': 'manufacturer', 'type': 'String'},
        {'field': 'model', 'type': 'String'},
        {'field': 'year', 'type': 'String'},
        {'field': 'mileage', 'type': 'String'},
        {'field': 'fuel_type', 'type': 'String'},
        {'field': 'transmission', 'type': 'String'},
        {'field': 'body_type', 'type': 'String'},
        {'field': 'cylinders', 'type': 'String'},
        {'field': 'drive', 'type': 'String'},
        {'field': 'color', 'type': 'String'},
    ]

    linker = dedupe.RecordLink(fields)

    return linker


def dedupe_training(ground_truth):

    labeled_pairs = couples_to_python_dictionary(ground_truth)
    linker = fields_to_dedupe_format()

    linker.markPairs(labeled_pairs)
    linker.train()

    return linker