from pathlib import Path

from src.Preprocessing.create_splits import splits_from_index_lists
from src.utils.load_splits import Load_Splits
from copy import deepcopy


def pretraining_finetuning(split_paths, db_names, pretraining_ids, finetuning_ids, appendices=None, output_path=None):
    """
    Combine split files for pretraining. The function reads multiple split files, recomputes the graph ids based on the order of the files
    and combines the files into a single output file using the specified pretraining_ids for pretraining and the remaining ids for finetuning.
    :param split_paths: List of paths to the split files.
    :param db_names: List of database names corresponding to the split files (must match the order of split_paths).
    :param appendices: List of appendices, if not using the default "_splits" for the split files.
    :param pretraining_ids: Positions of the split files in the list that should be used for pretraining.
    :param finetuning_ids: Positions of the split files in the list that should be used for finetuning.
    :param output_path_pretraining: Path to the output file where the pretraining graph ids will be written. If None, the default path will be used.
    :param output_path_finetuning: Path to the output file where the finetuning graph ids will be written. If None, the default path will be used.
    """
    updated_ids_splits = []
    if appendices is None:
        appendices = [None] * len(split_paths)  # Default to no appendix if not provided
    for split_path, db_name, appendix in zip(split_paths, db_names, appendices):
        splits = Load_Splits(split_path, db_name, appendix)
        updated_ids_splits.append(splits)
    # get max id in updated_ids_splits[0]
    max_id = 0
    test, train, vali = updated_ids_splits[0]
    for ids in test:
        max_id = max(max_id, max(ids))
    for ids in train:
        max_id = max(max_id, max(ids))
    for ids in vali:
        max_id = max(max_id, max(ids))

    # add the max_id to the ids in updated_ids_splits
    for i in range(1, len(updated_ids_splits)):
        test, train, vali = updated_ids_splits[i]
        new_test = []
        new_train = []
        new_vali = []
        for ids in test:
            new_test.append([x + max_id + 1 for x in ids])
        for ids in train:
            new_train.append([x + max_id + 1 for x in ids])
        for ids in vali:
            new_vali.append([x + max_id + 1 for x in ids])
        updated_ids_splits[i] = new_test, new_train, new_vali



    # generate pretraining splits from the combined splits
    # check wether there is no overlap between the pretraining and finetuning ids
    if not pretraining_ids:
        raise ValueError("pretraining_ids must not be empty")
    if not finetuning_ids:
        raise ValueError("finetuning_ids must not be empty")
    if set(pretraining_ids).intersection(set(finetuning_ids)):
        raise ValueError("pretraining_ids and finetuning_ids must not overlap")

    pretraining_splits = deepcopy(updated_ids_splits[min(pretraining_ids)])
    finetuning_splits = deepcopy(updated_ids_splits[min(finetuning_ids)])
    merged_splits = deepcopy(updated_ids_splits[0])

    for i in pretraining_ids:
        if i != min(pretraining_ids):
            test, train, vali = updated_ids_splits[i]
            for j in range(len(test)):
                pretraining_splits[0][j].extend(test[j])
            for j in range(len(train)):
                pretraining_splits[1][j].extend(train[j])
            for j in range(len(vali)):
                pretraining_splits[2][j].extend(vali[j])



    # generate finetuning splits
    for i in finetuning_ids:
        if i != min(finetuning_ids):
            test, train, vali = updated_ids_splits[i]
            for j in range(len(test)):
                finetuning_splits[0][j].extend(test[j])
            for j in range(len(train)):
                finetuning_splits[1][j].extend(train[j])
            for j in range(len(vali)):
                finetuning_splits[2][j].extend(vali[j])

    # generate merged splits
    # append all other splits to the merged splits
    for i in range(1, len(updated_ids_splits)):
        test, train, vali = updated_ids_splits[i]
        for j in range(len(test)):
            merged_splits[0][j].extend(test[j])
        for j in range(len(train)):
            merged_splits[1][j].extend(train[j])
        for j in range(len(vali)):
            merged_splits[2][j].extend(vali[j])


    # write the pretraining splits to the output file
    if output_path is None:
        output_path = split_paths[0]
    db_string = '_'.join(db_name for db_name in db_names)
    pretraining_string = '_pretraining_'
    finetuning_string = '_finetuning_'
    merged_string = '_merged'
    pretraining_string += '_'.join(db_names[i] for i in pretraining_ids)
    finetuning_string += '_'.join(db_names[i] for i in finetuning_ids)


    # write splits as a json file
    splits_from_index_lists(pretraining_splits[1], pretraining_splits[2], pretraining_splits[0], db_name=db_string + pretraining_string, output_path=output_path)
    splits_from_index_lists(finetuning_splits[1], finetuning_splits[2], finetuning_splits[0], db_name=db_string + finetuning_string, output_path=output_path)
    splits_from_index_lists(merged_splits[1], merged_splits[2], merged_splits[0], db_name=db_string + merged_string, output_path=output_path)
    pass
