from src.Preprocessing.create_splits import splits_from_index_lists


def zinc_splits(output_path, *args, **kwargs):
    training_indices = [list(range(0, 10000))]
    validation_indices = [list(range(10000, 11000))]
    test_indices = [list(range(11000, 12000))]
    return splits_from_index_lists(training_indices, validation_indices, test_indices, 'ZINC', output_path)

def planetoid_splits(output_path, graph_data, *args, **kwargs):
    # get all true values in the train mask
    training_indices = [graph_data.data.train_mask.nonzero().squeeze().tolist()]
    validation_indices = [graph_data.data.val_mask.nonzero().squeeze().tolist()]
    test_indices = [graph_data.data.test_mask.nonzero().squeeze().tolist()]
    return splits_from_index_lists(training_indices, validation_indices, test_indices, graph_data.name, output_path)