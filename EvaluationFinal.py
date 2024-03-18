import os

import pandas as pd
from matplotlib import pyplot as plt
from sympy import pretty_print as pp, latex
from sympy.abc import a, b, n

def epoch_accuracy(db_name, y_val, ids):
    if y_val == 'Train':
        y_val = 'EpochAccuracy'
    elif y_val == 'Validation':
        y_val = 'ValidationAccuracy'
    elif y_val == 'Test':
        y_val = 'TestAccuracy'


    # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
    # ge all those files
    files = []
    network_files = []
    for file in os.listdir(f"Results/{db_name}/Results"):
        for id in ids:
            id_str = str(id).zfill(6)
            # file contains the id_str
            if id_str in file:
                if file.startswith(f"{db_name}_") and file.endswith(".csv"):
                    files.append(file)
                elif file.startswith(f"{db_name}_") and file.endswith(".txt"):
                    network_files.append(file)
    df_all = None
    for i, file in enumerate(files):
        # get file id
        file_id = file.split('_')[1]
        df = pd.read_csv(f"Results/{db_name}/Results/{file}", delimiter=";")
        # add the file id to the dataframe
        df['FileId'] = file_id
        # concatenate the dataframes
        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=True)
    # open network file and read the network
    network_legend = {}
    for i, file in enumerate(network_files):
        with open(f"Results/{db_name}/Results/{file}", "r") as f:
            # get first line
            line = f.readline()
            # get string between [ and ]
            line = line[line.find("[")+1:line.find("]")]
            # split by , not in ''
            line = line.split(", ")
            k = line[1].split("_")[1].split(":")[0]
            d = 0
            if ":" in line[1]:
                d = len(line[1].split("_")[1].split(":")[1].split(","))
            bound = line[0]
            # cound number of occurrences of "wl" in line
            L = sum([1 for i in line if "wl" in i]) -1


            # remove last element
            line = line[:-1]
            # join the strings with ;
            line = ";".join(line)
            id = file.split('_')[1]
            # remove ' from k,d,bound and L
            k = k.replace("'", "")
            bound = bound.replace("'", "")
            if d == 0:
                # replace d and bound by '-'
                d = '-'
                bound = '-'
            if k == '20':
                k= 'max'
            #network_legend[id] = f'Id:{id}, {line}'
            char = '\u00b2'
            if L == 0:
                char = ''
            elif L == 1:
                char = '\u00b9'
            elif L == 2:
                char = '\u00b2'
            elif L == 3:
                char = '\u00b3'
            network_legend[id] = f'({k},{d},{bound}){char}'

    # group by file id
    groups = df_all.groupby('FileId')
    # for each group group by epoch and get the mean and std
    for i, id in enumerate(ids):
        id_str = str(id).zfill(6)
        group = groups.get_group(id_str)
        group_mean = group.groupby('Epoch').mean()
        group_std = group.groupby('Epoch').std()
        # plot the EpochAccuracy vs Epoch
        if i == 0:
            ax = group_mean.plot(y=y_val, yerr=group_std[y_val], label=network_legend[id_str])
        else:
            group_mean.plot(y=y_val, yerr=group_std[y_val], ax=ax, label=network_legend[id_str])

    # save to tikz
    #tikzplotlib.save(f"Results/{db_name}/Plots/{y_val}.tex")
    # set the title
    # two columns for the legend
    plt.legend(ncol=2)
    plt.title(f"{db_name}")
    # set y-axis from 0 to 100
    plt.ylim(0, 100)
    plt.savefig(f"Results/{db_name}/Plots/{db_name}_{y_val}.png")
    plt.show()

def evaluateGraphLearningNN(db_name, ids):
    evaluation = {}
    for id in ids:
        id_str = str(id).zfill(6)
        # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
        # ge all those files
        files = []
        network_files = []
        for file in os.listdir(f"Results/{db_name}/Results"):
            if file.startswith(f"{db_name}_{id_str}_Results_run_id_") and file.endswith(".csv"):
                files.append(file)
            elif file.startswith(f"{db_name}_{id_str}_Network") and file.endswith(".txt"):
                network_files.append(file)

        df_all = None
        for i, file in enumerate(files):
            df = pd.read_csv(f"Results/{db_name}/Results/{file}", delimiter=";")
            # concatenate the dataframes
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)

        # create a new column RunNumberValidationNumber that is the concatenation of RunNumber and ValidationNumber
        df_all['RunNumberValidationNumber'] = df_all['RunNumber'].astype(str) + df_all['ValidationNumber'].astype(str)

        # group the data by RunNumberValidationNumber
        groups = df_all.groupby('RunNumberValidationNumber')

        run_groups = df_all.groupby('RunNumber')
        # plot each run
        # for name, group in run_groups:
        #    group['TestAccuracy'].plot()
        # plt.show()

        indices = []
        # iterate over the groups
        for name, group in groups:
            # get the maximum validation accuracy
            max_val_acc = group['ValidationAccuracy'].max()
            # get the row with the maximum validation accuracy
            max_row = group[group['ValidationAccuracy'] == max_val_acc]
            # get the last row of max_row
            max_row = max_row.iloc[-1]
            # get the index of the row
            index = max_row.name
            indices.append(index)

        # get the rows with the indices
        df_validation = df_all.loc[indices]
        mean_validation = df_validation.mean()
        std_validation = df_validation.std()
        #print epoch accuracy
        print(f"Id: {id} Average Epoch Accuracy: {mean_validation['EpochAccuracy']} +/- {std_validation['EpochAccuracy']}")
        print(f"Id: {id} Average Validation Accuracy: {mean_validation['ValidationAccuracy']} +/- {std_validation['ValidationAccuracy']}")
        # if name is NCI1, then group by the ValidationNumber
        if db_name == 'NCI1' or db_name == 'ENZYMES' or db_name == 'PROTEINS' or db_name == 'DD' or db_name == 'IMDB-BINARY':
            df_validation = df_validation.groupby('ValidationNumber').mean()
        else:
            df_validation = df_validation.groupby('RunNumber').mean()
        # get the average and deviation over all runs
        avg = df_validation.mean()
        std = df_validation.std()
        # print the avg and std achieved by the highest validation accuracy
        print(f"Id: {id} Average Test Accuracy: {avg['TestAccuracy']} +/- {std['TestAccuracy']}")

        # open network file and read the network
        network_legend = {}
        with open(f"Results/{db_name}/Results/{network_files[0]}", "r") as f:
            # get first line
            line = f.readline()
            # get string between [ and ]
            line = line[line.find("[") + 1:line.find("]")]
            # split by , not in ''
            line = line.split(", ")
            # join the strings with ;
            line = ";".join(line)
            id = file.split('_')[1]
            network_legend[id] = f'Id:{id}, {line}'
        evaluation[id] = [avg['TestAccuracy'], std['TestAccuracy'], mean_validation['ValidationAccuracy'], std_validation['ValidationAccuracy'], network_legend[id]]

    # print all evaluation items start with id and network then validation and test accuracy
    # round all floats to 2 decimal places
    for key, value in evaluation.items():
        value[0] = round(value[0], 2)
        value[1] = round(value[1], 2)
        value[2] = round(value[2], 2)
        value[3] = round(value[3], 2)
        print(f"{value[4]} Validation Accuracy: {value[2]} +/- {value[3]} Test Accuracy: {value[0]} +/- {value[1]}")

    # print the evaluation items with the k highest validation accuracies
    print(f"Top 5 Validation Accuracies for {db_name}")
    k = 5
    sorted_evaluation = sorted(evaluation.items(), key=lambda x: x[1][2], reverse=True)
    for i in range(min(k, len(sorted_evaluation))):
        print(f"{sorted_evaluation[i][1][4]} Validation Accuracy: {sorted_evaluation[i][1][2]} +/- {sorted_evaluation[i][1][3]} Test Accuracy: {sorted_evaluation[i][1][0]} +/- {sorted_evaluation[i][1][1]}")


def main():
    evaluateGraphLearningNN(db_name='DHFR', ids=[1] + [i for i in range(4, 27)])
    evaluateGraphLearningNN(db_name='NCI1', ids=[i for i in range(4,23)] + [i for i in range(24, 26)] + [i for i in range(106, 117)])
    evaluateGraphLearningNN(db_name='ENZYMES', ids=[1])
    evaluateGraphLearningNN(db_name='PROTEINS', ids=[1])
    evaluateGraphLearningNN(db_name='IMDB-BINARY', ids=[1])
    evaluateGraphLearningNN(db_name='IMDB-MULTI', ids=[1])
    # epoch_accuracy(db_name='DHFR', y_val='Train', ids=[12,13,20,1,10,7,24,9,11,25])
    # epoch_accuracy(db_name='NCI1', y_val='Test', ids=[10,24,116,4,8,9,114,110,107,25])
    # epoch_accuracy(db_name='ENZYMES', y_val='Train', ids=[1])
    # epoch_accuracy(db_name='PROTEINS', y_val='Train', ids=[1])
    # epoch_accuracy(db_name='IMDB-BINARY', y_val='Train', ids=[1])
    # epoch_accuracy(db_name='ENZYMES', y_val='Test', ids=[1])
    # epoch_accuracy(db_name='PROTEINS', y_val='Test', ids=[1])
    # epoch_accuracy(db_name='IMDB-BINARY', y_val='Test', ids=[1])
    # epoch_accuracy(db_name='IMDB-MULTI', y_val='Test', ids=[1])
if __name__ == "__main__":
    main()