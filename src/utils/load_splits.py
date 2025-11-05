import json
import pickle


def Load_Splits(path):
    """
    Load the splits for a given database.
    :param path: Path to the splits file.
    :return: A dictionary containing the train, validation and test splits.
    """


    with open(path, "rb") as f:
        splits = json.load(f)

    test_indices = [x['test'] for x in splits]
    train_indices = [x['model_selection'][0]['train'] for x in splits]
    vali_indices = [x['model_selection'][0]['validation'] for x in splits]

    return {'test': test_indices, 'train': train_indices, 'validation': vali_indices}