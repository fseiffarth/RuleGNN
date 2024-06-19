import json
import pickle


def Load_Splits(path, db_name):
    splits = None
    with open(f"{path}/{db_name}_splits.json", "rb") as f:
        splits = json.load(f)

    test_indices = [x['test'] for x in splits]
    train_indices = [x['model_selection'][0]['train'] for x in splits]
    vali_indices = [x['model_selection'][0]['validation'] for x in splits]

    return test_indices, train_indices, vali_indices


if __name__ == "__main__":
    splits = Load_Splits("NCI1")
    print(splits)