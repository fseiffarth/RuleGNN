from src.Preprocessing.create_splits import splits_from_index_lists


def zinc_splits(output_path):
    training_indices = [list(range(0, 10000))]
    validation_indices = [list(range(10000, 11000))]
    test_indices = [list(range(11000, 12000))]
    return splits_from_index_lists(training_indices, validation_indices, test_indices, 'ZINC', output_path)