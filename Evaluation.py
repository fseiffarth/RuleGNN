import os

import pandas as pd
from matplotlib import pyplot as plt


def evaluateNoGKernel(db_name):
    # load the data from Results/{db_name}/Results/NoGKernel/Results.csv as a pandas dataframe
    df = pd.read_csv(f"Results/{db_name}/Results/NoGKernel/ResultsNN.csv")
    # get the average and deviation over all runs
    avg = df.mean()
    std = df.std()
    # print the average and deviation of the test accuracy
    print(f"Average Test Accuracy: {avg['Test Accuracy']} +/- {std['Test Accuracy']}")


def evaluateGraphLearningNN(db_name, id):
    method = 'highest_epoch'
    id_str = str(id).zfill(6)
    # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
    # ge all those files
    files = []
    for file in os.listdir(f"Results/{db_name}/Results"):
        if file.startswith(f"{db_name}_{id_str}_Results_run_id_") and file.endswith(".csv"):
            files.append(file)

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
    # group by the run
    df_validation = df_validation.groupby('RunNumber').mean()
    # get the average and deviation over all runs
    avg = df_validation.mean()
    std = df_validation.std()
    # print the avg and std achieved by the highest validation accuracy
    print(f"Average Test Accuracy: {avg['TestAccuracy']} +/- {std['TestAccuracy']}")

    # group the data by Epoch and add the standard deviation
    df_std = df_all.groupby('Epoch').std()
    df_all = df_all.groupby('Epoch').mean()

    # get the row with the highest validation accuracy
    max_val_acc = df_all['ValidationAccuracy'].max()
    max_row = df_all[df_all['ValidationAccuracy'] == max_val_acc]
    # get the index of the last row
    max_row = max_row.iloc[-1]
    # get the TestAccuracy of the row
    test_acc = max_row['TestAccuracy']
    # print the test accuracy of the highest validation accuracy
    print(f"Test Accuracy of highest Validation Accuracy: {test_acc}")


    df_all['TestStd'] = df_std['TestAccuracy']
    # plot the test accuracy over the epochs
    df_all['TestAccuracy'].plot()
    # plot the epoch accuracy over the epochs
    df_all['EpochAccuracy'].plot()
    # plot the validation accuracy over the epochs
    df_all['ValidationAccuracy'].plot()
    # add legend
    plt.legend(['Test Accuracy', 'Epoch Accuracy', 'Validation Accuracy'])
    plt.show()

    # get row as df with the highest test accuracy
    max_row = df_all[df_all['TestAccuracy'] == df_all['TestAccuracy'].max()]
    max_epoch = df_all[df_all['EpochAccuracy'] == df_all['EpochAccuracy'].max()]

    # create a pandas dataframe for each file and concatenate them
    df_all = None
    for i, file in enumerate(files):
        df = pd.read_csv(f"Results/{db_name}/Results/{file}", delimiter=";")
        if method == 'highest_epoch':
            # get all lines that have the highest epoch
            max_epoch = df['Epoch'].max()
            df = df[df['Epoch'] == max_epoch]
            # concatenate the dataframes
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)
        elif method == 'min_loss':
            # find the minimum epoch loss for each epoch grouped by ValidationNumber and mark the corresponding row
            min_loss = df.groupby('ValidationNumber')['EpochLoss'].idxmin()
            # get the rows in df that are determined by the column EpochLoss in min_loss
            df = df.iloc[min_loss]
            # concatenate the dataframes
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)

    # group the data by RunNumber and get the average and deviation over all runs
    df_all = df_all.groupby('RunNumber').mean()
    # get mean and std over df_all
    avg = df_all.mean()
    std = df_all.std()
    print(f"Average Test Accuracy: {avg['TestAccuracy']} +/- {std['TestAccuracy']}")
    print(f"Average Epoch Accuracy: {avg['EpochAccuracy']} +/- {std['EpochAccuracy']}")


def main():
    # evaluate the results of the no graph kernel
    # evaluateNoGKernel(db_name='MUTAG')
    evaluateGraphLearningNN(db_name='DHFR', id=1)
if __name__ == "__main__":
    main()