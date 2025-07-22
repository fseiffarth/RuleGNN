from ogb.graphproppred import PygGraphPropPredDataset

from src.Preprocessing.create_splits import splits_from_index_lists


def zinc_splits(output_path, *args, **kwargs):
    training_indices = [list(range(0, 10000))]
    validation_indices = [list(range(10000, 11000))]
    test_indices = [list(range(11000, 12000))]
    return splits_from_index_lists(training_indices, validation_indices, test_indices, 'ZINC', output_path)

def zinc_splits_full(output_path, *args, **kwargs):
    training_indices = [list(range(0, 220011))]
    validation_indices = [list(range(220011, 220011 + 24445))]
    test_indices = [list(range(220011 + 24445, 220011 + 24445 + 5000))]
    return splits_from_index_lists(training_indices, validation_indices, test_indices, 'ZINC-full', output_path)


def planetoid_splits(output_path, graph_data, *args, **kwargs):
    # get all true values in the train mask
    training_indices = [graph_data.data.train_mask.nonzero().squeeze().tolist()]
    validation_indices = [graph_data.data.val_mask.nonzero().squeeze().tolist()]
    test_indices = [graph_data.data.test_mask.nonzero().squeeze().tolist()]
    return splits_from_index_lists(training_indices, validation_indices, test_indices, graph_data.name, output_path)


def ogb_molhiv_splits(output_path, *args, **kwargs):
    db_name = "ogbg-molhiv"
    dataset_ogb = PygGraphPropPredDataset(name='ogbg-molhiv', root='tmp/')
    split_idx = dataset_ogb.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    return splits_from_index_lists([train_idx.tolist()], [valid_idx.tolist()], [test_idx.tolist()], db_name, output_path)


def ogb_splits(output_path, db_name, *args, **kwargs):
    dataset_ogb = PygGraphPropPredDataset(name=db_name, root='tmp/')
    split_idx = dataset_ogb.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    return splits_from_index_lists([train_idx.tolist()], [valid_idx.tolist()], [test_idx.tolist()], db_name, output_path)

def qm_splits(output_path, db_name, *args, **kwargs):
    training_indices = [list(range(0, 10000))]
    validation_indices = [list(range(10000, 11000))]
    test_indices = [list(range(11000, 12000))]
    return splits_from_index_lists(training_indices, validation_indices, test_indices, db_name, output_path)

def substructure_counting_splits(output_path, db_name, *args, **kwargs):
    training_indices = [list(range(0, 1500))]
    validation_indices = [list(range(1500, 2500))]
    test_indices = [list(range(2500, 5000))]
    return splits_from_index_lists(training_indices, validation_indices, test_indices, db_name, output_path)