import os

import pandas as pd
from matplotlib import pyplot as plt


def epoch_accuracy(db_name, y_val, ids):
    if y_val == 'Train':
        y_val = 'EpochAccuracy'
        size = 'TrainingSize'
    elif y_val == 'Validation':
        y_val = 'ValidationAccuracy'
        size = 'ValidationSize'
    elif y_val == 'Test':
        y_val = 'TestAccuracy'
        size = 'TestSize'

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
            line = line[line.find("[") + 1:line.find("]")]
            simple = True
            if simple:
                # split by ,
                line = line.split(", ")
                # join the strings
                id = file.split('_')[1]
                network_legend[id] = f'Id:{id}, {"".join(line)}'
            else:
                # split by , not in ''
                line = line.split(", ")
                k = line[1].split("_")[1].split(":")[0]
                d = 0
                if ":" in line[1]:
                    d = len(line[1].split("_")[1].split(":")[1].split(","))
                bound = line[0]
                # cound number of occurrences of "wl" in line
                L = sum([1 for i in line if "wl" in i]) - 1

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
                    k = 'max'
                # network_legend[id] = f'Id:{id}, {line}'
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
        group = groups.get_group(id_str).copy()
        group['EpochAccuracy'] = group['EpochAccuracy'] * group['TrainingSize']
        group['EpochLoss'] *= group['TrainingSize']
        group['ValidationAccuracy'] *= group['ValidationSize']
        group['ValidationLoss'] *= group['ValidationSize']
        group['TestAccuracy'] *= group['TestSize']
        # f = lambda x: sum(x['TrainingSize'] * x['EpochAccuracy']) / sum(x['TrainingSize'])

        group_mean = group.groupby('Epoch').mean(numeric_only=True)
        group_mean['EpochAccuracy'] /= group_mean['TrainingSize']
        group_mean['EpochLoss'] /= group_mean['TrainingSize']
        group_mean['ValidationAccuracy'] /= group_mean['ValidationSize']
        group_mean['ValidationLoss'] /= group_mean['ValidationSize']
        group_mean['TestAccuracy'] /= group_mean['TestSize']
        group_std = group.groupby('Epoch').std(numeric_only=True)
        group_std['EpochAccuracy'] /= group_mean['TrainingSize']
        group_std['EpochLoss'] /= group_mean['TrainingSize']
        group_std['ValidationAccuracy'] /= group_mean['ValidationSize']
        group_std['ValidationLoss'] /= group_mean['ValidationSize']
        group_std['TestAccuracy'] /= group_mean['TestSize']
        # plot the EpochAccuracy vs Epoch
        if i == 0:
            ax = group_mean.plot(y=y_val, yerr=group_std[y_val], label=network_legend[id_str])
        else:
            group_mean.plot(y=y_val, yerr=group_std[y_val], ax=ax, label=network_legend[id_str])

    # save to tikz
    # tikzplotlib.save(f"Results/{db_name}/Plots/{y_val}.tex")
    # set the title
    # two columns for the legend
    plt.legend(ncol=2)
    plt.title(f"{db_name}, {y_val}")
    # set y-axis from 0 to 100
    plt.ylim(0, 100)
    plt.savefig(f"Results/{db_name}/Plots/{db_name}_{y_val}.png")
    plt.show()


def evaluateGraphLearningNN(db_name, ids, path='Results/'):
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
            # get the minimum validation loss if column exists
            #if 'ValidationLoss' in group.columns:
            #    max_val_acc = group['ValidationLoss'].min()
            #    max_row = group[group['ValidationLoss'] == max_val_acc]

            # get row with the minimum validation loss
            min_val_loss = max_row['ValidationLoss'].min()
            max_row = group[group['ValidationLoss'] == min_val_loss]
            max_row = max_row.iloc[-1]
            # get the index of the row
            index = max_row.name
            indices.append(index)

        # get the rows with the indices
        df_validation = df_all.loc[indices]
        mean_validation = df_validation.mean(numeric_only=True)
        std_validation = df_validation.std(numeric_only=True)
        # print epoch accuracy
        print(
            f"Id: {id} Average Epoch Accuracy: {mean_validation['EpochAccuracy']} +/- {std_validation['EpochAccuracy']}")
        print(
            f"Id: {id} Average Validation Accuracy: {mean_validation['ValidationAccuracy']} +/- {std_validation['ValidationAccuracy']}")
        # if name is NCI1, then group by the ValidationNumber
        if db_name == 'NCI1' or db_name == 'ENZYMES' or db_name == 'PROTEINS' or db_name == 'DD' or db_name == 'IMDB-BINARY' or db_name == 'IMDB-MULTI' or db_name == "SYNTHETICnew" or db_name == "DHFR" or db_name == "NCI109" or db_name == "Mutagenicity" or db_name == "MUTAG":
            df_validation = df_validation.groupby('ValidationNumber').mean(numeric_only=True)
        else:
            df_validation = df_validation.groupby('RunNumber').mean(numeric_only=True)
        # get the average and deviation over all runs

        df_validation['EpochLoss'] *= df_validation['TrainingSize']
        df_validation['TestAccuracy'] *= df_validation['TestSize']
        df_validation['ValidationAccuracy'] *= df_validation['ValidationSize']
        df_validation['ValidationLoss'] *= df_validation['ValidationSize']
        avg = df_validation.mean(numeric_only=True)

        avg['EpochLoss'] /= avg['TrainingSize']
        avg['TestAccuracy'] /= avg['TestSize']
        avg['ValidationAccuracy'] /= avg['ValidationSize']
        avg['ValidationLoss'] /= avg['ValidationSize']

        std = df_validation.std(numeric_only=True)
        std['EpochLoss'] /= avg['TrainingSize']
        std['TestAccuracy'] /= avg['TestSize']
        std['ValidationAccuracy'] /= avg['ValidationSize']
        std['ValidationLoss'] /= avg['ValidationSize']

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
        # check if ValidationLoss exists
        if 'ValidationLoss' in df_all.columns:
            evaluation[id] = [avg['TestAccuracy'], std['TestAccuracy'], mean_validation['ValidationAccuracy'],
                              std_validation['ValidationAccuracy'], network_legend[id],
                              mean_validation['ValidationLoss'], std_validation['ValidationLoss']]
        else:
            evaluation[id] = [avg['TestAccuracy'], std['TestAccuracy'], mean_validation['ValidationAccuracy'],
                              std_validation['ValidationAccuracy'], network_legend[id]]

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
        if len(sorted_evaluation[i][1]) > 5:
            sorted_evaluation = sorted(sorted_evaluation, key=lambda x: x[1][5], reverse=False)
            print(
                f"{sorted_evaluation[i][1][4]} Validation Loss: {sorted_evaluation[i][1][5]} +/- {sorted_evaluation[i][1][6]} Validation Accuracy: {sorted_evaluation[i][1][2]} +/- {sorted_evaluation[i][1][3]} Test Accuracy: {sorted_evaluation[i][1][0]} +/- {sorted_evaluation[i][1][1]}")
        else:
            print(
                f"{sorted_evaluation[i][1][4]} Validation Accuracy: {sorted_evaluation[i][1][2]} +/- {sorted_evaluation[i][1][3]} Test Accuracy: {sorted_evaluation[i][1][0]} +/- {sorted_evaluation[i][1][1]}")


def model_selection_evaluation(db_name, path='Results', ids=None):
    evaluation = {}

    if ids is None:
        # get all ids
        ids = []
        for file in os.listdir(f"{path}/{db_name}/Results"):
            if file.endswith(".txt"):
                id = int(file.split('_')[-2])
                ids.append(id)
        # sort the ids
        ids.sort()

    for id in ids:
        id_str = str(id).zfill(6)
        # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
        # ge all those files
        files = []
        network_files = []
        for file in os.listdir(f"{path}/{db_name}/Results"):
            if file.startswith(f"{db_name}_Configuration_{id_str}_Results_run_id_") and file.endswith(".csv"):
                files.append(file)
            elif file.startswith(f"{db_name}_Configuration_{id_str}_Network") and file.endswith(".txt"):
                network_files.append(file)

        df_all = None
        for i, file in enumerate(files):
            df = pd.read_csv(f"{path}/{db_name}/Results/{file}", delimiter=";")
            # concatenate the dataframes
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)

        # create a new column RunNumberValidationNumber that is the concatenation of RunNumber and ValidationNumber
        df_all['RunNumberValidationNumber'] = df_all['RunNumber'].astype(str) + df_all['ValidationNumber'].astype(str)

        # group the data by RunNumberValidationNumber
        groups = df_all.groupby('RunNumberValidationNumber')

        indices = []
        # iterate over the groups
        for name, group in groups:
            # get the maximum validation accuracy
            max_val_acc = group['ValidationAccuracy'].max()
            # get the row with the maximum validation accuracy
            max_row = group[group['ValidationAccuracy'] == max_val_acc]
            # get the minimum validation loss if column exists
            #if 'ValidationLoss' in group.columns:
            #    max_val_acc = group['ValidationLoss'].min()
            #    max_row = group[group['ValidationLoss'] == max_val_acc]

            # get row with the minimum validation loss
            min_val_loss = max_row['ValidationLoss'].min()
            max_row = group[group['ValidationLoss'] == min_val_loss]
            max_row = max_row.iloc[-1]
            # get the index of the row
            index = max_row.name
            indices.append(index)

        # get the rows with the indices
        df_validation = df_all.loc[indices]
        df_validation = df_validation.groupby('ValidationNumber').mean(numeric_only=True)

        # get the average and deviation over all runs

        df_validation['EpochLoss'] *= df_validation['TrainingSize']
        df_validation['TestAccuracy'] *= df_validation['TestSize']
        df_validation['TestLoss'] *= df_validation['TestSize']
        df_validation['ValidationAccuracy'] *= df_validation['ValidationSize']
        df_validation['ValidationLoss'] *= df_validation['ValidationSize']
        avg = df_validation.mean(numeric_only=True)

        avg['EpochLoss'] /= avg['TrainingSize']
        avg['TestAccuracy'] /= avg['TestSize']
        avg['TestLoss'] /= avg['TestSize']
        avg['ValidationAccuracy'] /= avg['ValidationSize']
        avg['ValidationLoss'] /= avg['ValidationSize']

        std = df_validation.std(numeric_only=True)
        std['EpochLoss'] /= avg['TrainingSize']
        std['TestAccuracy'] /= avg['TestSize']
        std['TestLoss'] /= avg['TestSize']
        std['ValidationAccuracy'] /= avg['ValidationSize']
        std['ValidationLoss'] /= avg['ValidationSize']

        # print the avg and std achieved by the highest validation accuracy
        print(f"Id: {id} Average Test Accuracy: {avg['TestAccuracy']} +/- {std['TestAccuracy']}")


        evaluation[id] = [avg['TestAccuracy'], std['TestAccuracy'], avg['ValidationAccuracy'],
                              std['ValidationAccuracy'],
                              avg['ValidationLoss'], std['ValidationLoss'],
                          avg['TestLoss'], std['TestLoss'],
                          avg['EpochLoss'], std['EpochLoss']]


    # print all evaluation items start with id and network then validation and test accuracy
    # round all floats to 2 decimal places
    for key, value in evaluation.items():
        value[0] = round(value[0], 4)
        value[1] = round(value[1], 4)
        value[2] = round(value[2], 4)
        value[3] = round(value[3], 4)
        value[4] = round(value[4], 4)
        value[5] = round(value[5], 4)
        value[6] = round(value[6], 4)
        value[7] = round(value[7], 4)
        value[8] = round(value[8], 4)
        value[9] = round(value[9], 4)

    # print the evaluation items with the k highest validation accuracies
    print(f"Top 5 Validation Accuracies for {db_name}")
    k = 10

    sort_key = 2
    reversed_sort = True
    if db_name == 'ZINC':
        sort_key = 8
        reversed_sort = False

    sorted_evaluation = sorted(evaluation.items(), key=lambda x: x[1][sort_key], reverse=reversed_sort)

    for i in range(min(k, len(sorted_evaluation))):
        sorted_evaluation = sorted(sorted_evaluation, key=lambda x: x[1][sort_key], reverse=reversed_sort)
        print(
            f" Id: {sorted_evaluation[i][0]} "
            f" Epoch Loss: {sorted_evaluation[i][1][8]} +/- {sorted_evaluation[i][1][9]} "
            f" Validation Loss: {sorted_evaluation[i][1][4]} +/- {sorted_evaluation[i][1][5]} Validation Accuracy: {sorted_evaluation[i][1][2]} +/- {sorted_evaluation[i][1][3]} Test Accuracy: {sorted_evaluation[i][1][0]} +/- {sorted_evaluation[i][1][1]} Test Loss: {sorted_evaluation[i][1][6]} +/- {sorted_evaluation[i][1][7]}")


def best_model_evaluation(db_name, path='Results', ids=None):
    evaluation = {}

    if ids is None:
        # get all ids
        ids = []
        for file in os.listdir(f"{path}/{db_name}/Results"):
            if file.endswith(".txt") and "Best" in file:
                id = int(file.split('_')[-2])
                ids.append(id)
        # sort the ids
        ids.sort()

    for id in ids:
        id_str = str(id).zfill(6)
        # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
        # ge all those files
        files = []
        network_files = []
        for file in os.listdir(f"{path}/{db_name}/Results"):
            if file.startswith(f"{db_name}_Best_Configuration_{id_str}_Results_run_id_") and file.endswith(".csv"):
                files.append(file)
            elif file.startswith(f"{db_name}_Best_Configuration_{id_str}_Network") and file.endswith(".txt"):
                network_files.append(file)

        df_all = None
        for i, file in enumerate(files):
            df = pd.read_csv(f"{path}/{db_name}/Results/{file}", delimiter=";")
            # concatenate the dataframes
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)

        # create a new column RunNumberValidationNumber that is the concatenation of RunNumber and ValidationNumber
        df_all['RunNumberValidationNumber'] = df_all['RunNumber'].astype(str) + df_all['ValidationNumber'].astype(str)

        # group the data by RunNumberValidationNumber
        groups = df_all.groupby('RunNumberValidationNumber')

        indices = []
        # iterate over the groups
        for name, group in groups:
            # if db name is zinc, look for the minimum validation loss
            if db_name == 'ZINC':
                # get row with the minimum validation loss
                min_val_loss = group['ValidationLoss'].min()
                max_row = group[group['ValidationLoss'] == min_val_loss]
                max_row = max_row.iloc[-1]
                # get the index of the row
                index = max_row.name
                indices.append(index)
            else:
                # get the maximum validation accuracy
                max_val_acc = group['ValidationAccuracy'].max()
                # get the row with the maximum validation accuracy
                max_row = group[group['ValidationAccuracy'] == max_val_acc]
                # get the minimum validation loss if column exists
                #if 'ValidationLoss' in group.columns:
                #    max_val_acc = group['ValidationLoss'].min()
                #    max_row = group[group['ValidationLoss'] == max_val_acc]

                # get row with the minimum validation loss
                min_val_loss = max_row['ValidationLoss'].min()
                max_row = group[group['ValidationLoss'] == min_val_loss]
                max_row = max_row.iloc[-1]
                # get the index of the row
                index = max_row.name
                indices.append(index)

        # get the rows with the indices
        df_validation = df_all.loc[indices]
        df_validation = df_validation.groupby('ValidationNumber').mean(numeric_only=True)

        # get the average and deviation over all runs

        df_validation['EpochLoss'] *= df_validation['TrainingSize']
        df_validation['TestAccuracy'] *= df_validation['TestSize']
        df_validation['ValidationAccuracy'] *= df_validation['ValidationSize']
        df_validation['ValidationLoss'] *= df_validation['ValidationSize']
        avg = df_validation.mean(numeric_only=True)

        avg['EpochLoss'] /= avg['TrainingSize']
        avg['TestAccuracy'] /= avg['TestSize']
        avg['ValidationAccuracy'] /= avg['ValidationSize']
        avg['ValidationLoss'] /= avg['ValidationSize']

        std = df_validation.std(numeric_only=True)
        std['EpochLoss'] /= avg['TrainingSize']
        std['TestAccuracy'] /= avg['TestSize']
        std['ValidationAccuracy'] /= avg['ValidationSize']
        std['ValidationLoss'] /= avg['ValidationSize']

        # print the avg and std achieved by the highest validation accuracy
        print(f"Id: {id} Average Test Accuracy: {avg['TestAccuracy']} +/- {std['TestAccuracy']}")


        evaluation[id] = [avg['TestAccuracy'], std['TestAccuracy'], avg['ValidationAccuracy'],
                              std['ValidationAccuracy'],
                              avg['ValidationLoss'], std['ValidationLoss']]


    # print all evaluation items start with id and network then validation and test accuracy
    # round all floats to 2 decimal places
    for key, value in evaluation.items():
        value[0] = round(value[0], 4)
        value[1] = round(value[1], 4)
        value[2] = round(value[2], 4)
        value[3] = round(value[3], 4)
        value[4] = round(value[4], 4)
        value[5] = round(value[5], 4)
        #print(f"{value[4]} Validation Accuracy: {value[2]} +/- {value[3]} Test Accuracy: {value[0]} +/- {value[1]}")

    # print the evaluation items with the k highest validation accuracies
    print(f"Top 5 Validation Accuracies for {db_name}")
    k = 5
    sorted_evaluation = sorted(evaluation.items(), key=lambda x: x[1][2], reverse=True)

    for i in range(min(k, len(sorted_evaluation))):
        sorted_evaluation = sorted(sorted_evaluation, key=lambda x: x[1][2], reverse=True)
        print(
            f"Id: {sorted_evaluation[i][0]} Epoch Loss: Validation Loss: {sorted_evaluation[i][1][4]} +/- {sorted_evaluation[i][1][5]} Validation Accuracy: {sorted_evaluation[i][1][2]} +/- {sorted_evaluation[i][1][3]} Test Accuracy: {sorted_evaluation[i][1][0]} +/- {sorted_evaluation[i][1][1]}")


def main():
    #ids = [i for i in range(0, 51)]
    #final_evaluation(db_name='MUTAG', ids=ids)

    model_selection_evaluation(db_name='ZINC', path='RESULTS/Features')
    best_model_evaluation(db_name='IMDB-MULTI', path='RESULTS/Longrange')

    ids = [i for i in range(4, 7)]
    evaluateGraphLearningNN(db_name='SYNTHETICnew', ids=ids)
    epoch_accuracy(db_name='SYNTHETICnew', y_val='Train', ids=ids)
    epoch_accuracy(db_name='SYNTHETICnew', y_val='Validation', ids=ids)
    epoch_accuracy(db_name='SYNTHETICnew', y_val='Test', ids=ids)

    ids = [i for i in range(4, 7)]
    evaluateGraphLearningNN(db_name='DHFR', ids=ids)
    epoch_accuracy(db_name='DHFR', y_val='Train', ids=ids)
    epoch_accuracy(db_name='DHFR', y_val='Validation', ids=ids)
    epoch_accuracy(db_name='DHFR', y_val='Test', ids=ids)

    ids = [i for i in range(201, 205)]
    evaluateGraphLearningNN(db_name='NCI1', ids=ids)
    epoch_accuracy(db_name='NCI1', y_val='Train', ids=ids)
    epoch_accuracy(db_name='NCI1', y_val='Validation', ids=ids)
    epoch_accuracy(db_name='NCI1', y_val='Test', ids=ids)

    # Testing with MUTAG
    # ids = [i for i in range(7, 145)]
    # print_ids = [i for i in range(137, 150)]
    # evaluateGraphLearningNN(db_name='MUTAG', ids=ids)
    # evaluateGraphLearningNN(db_name='MUTAG', ids=print_ids)
    # epoch_accuracy(db_name='MUTAG', y_val='Train', ids=print_ids)
    # epoch_accuracy(db_name='MUTAG', y_val='Validation', ids=print_ids)
    # epoch_accuracy(db_name='MUTAG', y_val='Test', ids=print_ids)

    # epoch_accuracy(db_name='MUTAG', y_val='Train', ids=ids)
    # epoch_accuracy(db_name='MUTAG', y_val='Validation', ids=ids)
    # epoch_accuracy(db_name='MUTAG', y_val='Test', ids=ids)
    #
    # evaluateGraphLearningNN(db_name='DD', ids=[1])
    # evaluateGraphLearningNN(db_name='SYNTHETICnew', ids=[1,2,3])
    # evaluateGraphLearningNN(db_name='NCI109', ids=[1,2,3])
    # evaluateGraphLearningNN(db_name='Mutagenicity', ids=[2,3,4])
    # evaluateGraphLearningNN(db_name='DHFR', ids=[1,2,3])
    #
    # evaluateGraphLearningNN(db_name='DHFR', ids=[1] + [i for i in range(4, 27)])
    # evaluateGraphLearningNN(db_name='NCI1', ids=[i for i in range(4,23)] + [i for i in range(24, 26)] + [i for i in range(106, 117)])
    # evaluateGraphLearningNN(db_name='ENZYMES', ids=[1])
    # evaluateGraphLearningNN(db_name='PROTEINS', ids=[1])
    # evaluateGraphLearningNN(db_name='IMDB-BINARY', ids=[1])
    # evaluateGraphLearningNN(db_name='IMDB-MULTI', ids=[1,2,3])
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
