import json


def csl_to_json():
    # load the split data and convert it to the json split format
    test = []
    with open("CSL_test.txt", "r") as f:
        for line in f:
            ids = []
            for l in line.split(","):
                ids.append(int(l))
            test.append(ids)
    train = []
    with open("CSL_train.txt", "r") as f:
        for line in f:
            ids = []
            for l in line.split(","):
                ids.append(int(l))
            train.append(ids)
    validate = []
    with open("CSL_val.txt", "r") as f:
        for line in f:
            ids = []
            for l in line.split(","):
                ids.append(int(l))
            validate.append(ids)

    # create dictionary
    splits = []
    for i in range(len(test)):
        splits.append({"test" : test[i], "model_selection" : [{"train" : train[i], "validation" : validate[i]}]})

    # save splits to json as one line use json.dumps
    with open(f"CSL_splits.json", "w") as f:
        f.write(json.dumps(splits))


if __name__ == "__main__":
    csl_to_json()