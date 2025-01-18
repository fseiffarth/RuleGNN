import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def baseline_results(algorithm: str, datasets:list[str], path:str, sota:bool=False, first_column:str=''):
    '''
    get the results and print them in tabular form
    '''
    out_str = algorithm
    results = []
    # iterate over the datasets
    for dataset in datasets:
        dataset_path = f"{path}/{dataset}"
        # open and merge all csv files in dataset_path
        df_all = None
        for file in Path(dataset_path).iterdir():
            if file.suffix == '.csv':
                df = pd.read_csv(file, delimiter=";")
                # concatenate the dataframes
                if df_all is None:
                    df_all = df
                else:
                    df_all = pd.concat([df_all, df], ignore_index=True)
        # get groups from RunNumber and Algorithm
        df_results = []
        algorithm_groups = df_all.groupby('Algorithm')
        algorithm_mean_results = None
        for alg, algorithm_data in algorithm_groups:
            if alg == algorithm:
                run_group = algorithm_data.groupby('RunNumber')
                for run_number, group in run_group:
                    # remove algorithm and dataset columns
                    group = group.drop(columns=['Algorithm', 'Dataset'])
                    # multiply the TestAccuracy by 100
                    if not sota:
                        group['TestAccuracy'] = group['TestAccuracy'] * 100
                    else:
                        group['ValidationAccuracy'] = group['ValidationAccuracy'] * 100
                    # group each group by HyperparameterSVC and HyperparameterAlgo
                    grouped_results = group.groupby(['HyperparameterSVC', 'HyperparameterAlgo']).mean()
                    grouped_results_std = group.groupby(['HyperparameterSVC', 'HyperparameterAlgo']).std()
                    if not sota:
                        grouped_results['TestAccuracyStd'] = grouped_results_std['TestAccuracy']
                    else:
                        grouped_results['ValidationAccuracyStd'] = grouped_results_std['ValidationAccuracy']
                    grouped_results['RunNumber'] = run_number
                    if algorithm_mean_results is None:
                        algorithm_mean_results = grouped_results
                    else:
                        algorithm_mean_results = pd.concat([algorithm_mean_results, grouped_results], ignore_index=True)
                # get the mean of the algorithm_mean_results
                algorithm_mean_results = algorithm_mean_results.groupby(['RunNumber', 'HyperparameterSVC', 'HyperparameterAlgo']).mean()
                algorithm_mean_results['Dataset'] = dataset
                algorithm_mean_results['Algorithm'] = algorithm
                break
            else:
                continue

        # open summary_best_mean.csv
        # sort the algorithm_mean_results by ValidationAccuracy
        algorithm_mean_results = algorithm_mean_results.sort_values(by='ValidationAccuracy', ascending=False)
        data = algorithm_mean_results.iloc[0]
        if not sota:
            test_acc = data['TestAccuracy']
            test_std = data['TestAccuracyStd']
        else:
            test_acc = data['ValidationAccuracy']
            test_std = data['ValidationAccuracyStd']
        # round the test accuracy and test std to 2 decimal places
        test_acc = round(test_acc, 1)
        test_std = round(test_std, 1)
        results.append((test_acc, test_std))
        out_str += f' & ${test_acc} \\pm {test_std}$'


    return out_str, results


def share_gnn_results(algorithm: str, datasets:list[str], path:str, sota:bool=False):
    '''
    get the results and print them in tabular form
    '''
    out_str = algorithm
    results = []
    # iterate over the datasets
    for dataset in datasets:
        # open summary_best_mean.csv
        if not sota:
            df = pd.read_csv(f"{path}/{dataset}/summary_best_mean.csv", delimiter=",")
            test_acc = df['Test Accuracy Mean'].values[0]
            test_std = df['Test Accuracy Std'].values[0]
            # round the test accuracy and test std to 2 decimal places
            test_acc = round(test_acc, 1)
            test_std = round(test_std, 1)
            results.append((test_acc, test_std))
            out_str += f' & ${test_acc} \\pm {test_std}$'
            pass
        else:
            df = pd.read_csv(f"{path}/{dataset}/summary_sota.csv", delimiter=",")
            # get mean over configuration Id
            df = df.groupby('ConfigurationId').mean()
            # sort by Validation Accuracy
            df = df.sort_values(by='Validation Accuracy Mean', ascending=False)
            test_acc = df['Validation Accuracy Mean'].values[0]
            test_std = df['Validation Accuracy Std'].values[0]
            # round the test accuracy and test std to 2 decimal places
            test_acc = round(test_acc, 1)
            test_std = round(test_std, 1)
            results.append((test_acc, test_std))
            out_str += f' & ${test_acc} \\pm {test_std}$'
            pass
    return out_str, results

def fair_gnn_results(algorithm: str, datasets:list[str], path:str, first_column:str=''):
    '''
    get the results and print them in tabular form
    '''
    out_str = algorithm + first_column
    results = []
    # iterate over the datasets
    for dataset in datasets:
        if Path(f"{path}/{algorithm}_{dataset}_assessment/10_NESTED_CV/assessment_results.json").exists():
            with open(f"{path}/{algorithm}_{dataset}_assessment/10_NESTED_CV/assessment_results.json", 'r') as f:
                data = json.load(f)
                pass
            # open summary_best_mean.csv
            test_acc = data['avg_TS_score']
            test_std = data['std_TS_score']
            # round the test accuracy and test std to 2 decimal places
            test_acc = round(test_acc, 1)
            test_std = round(test_std, 1)
        elif Path(f"{path}/{algorithm}_{dataset}_assessment/5_NESTED_CV/assessment_results.json").exists():
            with open(f"{path}/{algorithm}_{dataset}_assessment/5_NESTED_CV/assessment_results.json", 'r') as f:
                data = json.load(f)
                pass
            # open summary_best_mean.csv
            test_acc = data['avg_TS_score']
            test_std = data['std_TS_score']
            # round the test accuracy and test std to 2 decimal places
            test_acc = round(test_acc, 1)
            test_std = round(test_std, 1)
        else:
            test_acc = 0
            test_std = 0
        results.append((test_acc, test_std))
        out_str += f' & ${test_acc} \\pm {test_std}$'
    return out_str, results

def print_table(first_columns, column_names, results, rules:list[int], with_colors:bool=True, with_std:bool=True, positive_negative_colors:bool=False):

    lines = []
    accuracies = results[:, :, :1].squeeze(2)
    stds = results[:, :, 1:].squeeze(2)

    # best three values per column
    best_column_values = np.array([np.sort(accuracies[:, i])[-3:] for i in range(accuracies.shape[1])])
    best_column_values = best_column_values.T

    if len(first_columns) != results.shape[0]:
        raise ValueError('The number of first columns and the number of rows in the results do not match')
    else:
        for i, first_column in enumerate(first_columns):
            line = first_column
            for j in range(accuracies.shape[1]):
                if accuracies[i, j] in best_column_values[:, j] and with_colors:
                    # get index of the value
                    index = np.where(best_column_values[:, j] == accuracies[i, j])[0][0]
                    if index == 0:
                        if with_std:
                            line += f' & \\ThirdColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                        else:
                            line += f' & \\ThirdColor{{{accuracies[i, j]}}}'
                    elif index == 1:
                        if with_std:
                            line += f' & \\SecondColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                        else:
                            line += f' & \\SecondColor{{{accuracies[i, j]}}}'
                    elif index == 2:
                        if with_std:
                            line += f' & \\FirstColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                        else:
                            line += f' & \\FirstColor{{{accuracies[i, j]}}}'
                else:
                    if positive_negative_colors:
                        if with_std:
                            if accuracies[i, j] > 0:
                                line += f' & \\FirstColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                            else:
                                line += f' & \\SecondColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                        else:
                            if accuracies[i, j] > 0:
                                line += f' & \\FirstColor{{{accuracies[i, j]}}}'
                            else:
                                line += f' & \\SecondColor{{{accuracies[i, j]}}}'
                    else:
                        if with_std:
                            line += f' & ${accuracies[i, j]} \\pm {stds[i, j]}$'
                        else:
                            line += f' & ${accuracies[i, j]}$'
            lines.append(line)
    # print the table in latex
    print('\\begin{tabular}{l' + 'c' * len(column_names) + '}')
    header = ' & '.join([''] + column_names) + '\\\\'
    print(header)
    print('\\toprule')
    for i,line in enumerate(lines):
        if i in rules:
            print('\\midrule')
        print(line + '\\\\')
    print('\\bottomrule')
    print('\\end{tabular}')

def fair_table():
    # print the large table
    datasets = ['NCI1', 'NCI109', 'Mutagenicity', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI']
    rows = []
    results = []

    baseline_algorithms = ['NoGKernel', 'WLKernel']
    baseline_first_columns = ['\\cite{Schulz2019OnTN}', '\\cite{DBLP:journals/jmlr/ShervashidzeSLMB11}']
    for i, algorithm in enumerate(baseline_algorithms):
        row_string, row_results = baseline_results(algorithm, datasets, 'Reproduce_RuleGNN/Results/RealWorld/Baseline/', first_column=baseline_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    fair_algorithms = ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'GATv2']
    fair_first_columns = ['\\cite{DBLP:conf/iclr/KipfW17}', '\\cite{Hamilton2017InductiveRL}', '\\cite{DBLP:conf/iclr/XuHLJ19}', '\\cite{Velickovic2017GraphAN}', '\\cite{DBLP:conf/iclr/Brody0Y22}']
    fair_path = 'Reproduce_RuleGNN/RESULTS/'
    for i, algorithm in enumerate(fair_algorithms):
        row_string, row_results = fair_gnn_results(algorithm, datasets, fair_path, first_column=fair_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    row, row_results = share_gnn_results('\\MyGNN (ours)', datasets, 'Reproduce_RuleGNN/Results/RealWorld/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Random (ours)', datasets, 'Reproduce_RuleGNN/Results/RealWorld/Random/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Encoder (ours)', datasets, 'Reproduce_RuleGNN/Results/RealWorld/Encoder/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Decoder (ours)', datasets, 'Reproduce_RuleGNN/Results/RealWorld/Decoder/')
    rows.append(row)
    results.append(row_results)
    # results to numpy array
    results = np.array(results)

    first_columns = ([f'{x} {baseline_first_columns[i]}' for i, x in enumerate(baseline_algorithms)]
                     + [f'{x} {fair_first_columns[i]}' for i, x in enumerate(fair_algorithms)]
                     + ['\\textbf{\\MyGNN (ours)}', '\\textbf{\\MyGNN-Random (ours)}', '\\textbf{\\MyGNN-Encoder (ours)}', '\\textbf{\\MyGNN-Decoder (ours)}'])

    print_table(first_columns, ['\\textbf{NCI1}', '\\textbf{NCI109}', '\\textbf{Mutagenicity}', '\\textbf{DHFR}', '\\textbf{IMDB-B}', '\\textbf{IMDB-M}']
                , results,
                [2, 7])

def fair_table_full():
    # print the large table
    datasets = ['NCI1', 'NCI109', 'Mutagenicity', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI']
    rows = []
    results = []

    baseline_algorithms = ['NoGKernel', 'WLKernel']
    baseline_first_columns = ['\\cite{Schulz2019OnTN}', '\\cite{DBLP:journals/jmlr/ShervashidzeSLMB11}']
    for i, algorithm in enumerate(baseline_algorithms):
        row_string, row_results = baseline_results(algorithm, datasets, 'Reproduce_RuleGNN/Results/RealWorld/Baseline/', first_column=baseline_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    fair_algorithms = ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'GATv2']
    fair_first_columns = ['\\cite{DBLP:conf/iclr/KipfW17}', '\\cite{Hamilton2017InductiveRL}', '\\cite{DBLP:conf/iclr/XuHLJ19}', '\\cite{Velickovic2017GraphAN}', '\\cite{DBLP:conf/iclr/Brody0Y22}']
    fair_path = 'Reproduce_RuleGNN/RESULTS/'
    for i, algorithm in enumerate(fair_algorithms):
        row_string, row_results = fair_gnn_results(algorithm, datasets, fair_path, first_column=fair_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    row, row_results = share_gnn_results('\\MyGNN (ours)', datasets, 'Reproduce_RuleGNN/Results/RealWorld/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Random (ours)', datasets, 'Reproduce_RuleGNN/Results/RealWorld/Random/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Encoder (ours)', datasets, 'Reproduce_RuleGNN/Results/RealWorld/Encoder/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Decoder (ours)', datasets, 'Reproduce_RuleGNN/Results/RealWorld/Decoder/')
    rows.append(row)
    results.append(row_results)
    # results to numpy array
    results = np.array(results)

    synthetic_datasets = ['LongRings100', 'EvenOddRingsCount16', 'EvenOddRings2_16', 'CSL', 'Snowflakes']
    synthetic_rows = []
    synthetic_results = []
    for i, algorithm in enumerate(baseline_algorithms):
        row_string, row_results = baseline_results(algorithm, synthetic_datasets, 'Reproduce_RuleGNN/Results/Synthetic/Baseline/', first_column=baseline_first_columns[i])
        synthetic_rows.append(row_string)
        synthetic_results.append(row_results)

    for i, algorithm in enumerate(fair_algorithms):
        row_string, row_results = fair_gnn_results(algorithm, synthetic_datasets, fair_path, first_column=fair_first_columns[i])
        synthetic_rows.append(row_string)
        synthetic_results.append(row_results)

    row, row_results = share_gnn_results('\\MyGNN (ours)', synthetic_datasets, 'Reproduce_RuleGNN/Results/Synthetic/')
    synthetic_rows.append(row)
    synthetic_results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Random (ours)', synthetic_datasets, 'Reproduce_RuleGNN/Results/Synthetic/Random/')
    synthetic_rows.append(row)
    synthetic_results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Encoder (ours)', synthetic_datasets, 'Reproduce_RuleGNN/Results/Synthetic/Encoder/')
    synthetic_rows.append(row)
    synthetic_results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Decoder (ours)', synthetic_datasets, 'Reproduce_RuleGNN/Results/Synthetic/Decoder/')
    synthetic_rows.append(row)
    synthetic_results.append(row_results)
    # results to numpy array
    synthetic_results = np.array(synthetic_results)
    # concatenate the results
    results = np.concatenate((results, synthetic_results), axis=1)


    first_columns = ([f'{x} {baseline_first_columns[i]}' for i, x in enumerate(baseline_algorithms)]
                     + [f'{x} {fair_first_columns[i]}' for i, x in enumerate(fair_algorithms)]
                     + ['\\textbf{\\MyGNN (ours)}', '\\textbf{\\MyGNN-Random (ours)}', '\\textbf{\\MyGNN-Encoder (ours)}', '\\textbf{\\MyGNN-Decoder (ours)}'])

    print_table(first_columns, ['\\textbf{NCI1}', '\\textbf{NCI109}', '\\textbf{Mutagen.}', '\\textbf{DHFR}', '\\textbf{IMDB-B}', '\\textbf{IMDB-M}',
                                '\\textbf{RingT1}', '\\textbf{RingT2}', '\\textbf{RingT3}', '\\textbf{CSL}', '\\textbf{Snowfl.}']
                , results,
                [2, 7])


def sota_baseline_and_share():
    datasets = ['NCI1', 'NCI109', 'IMDB-BINARY', 'IMDB-MULTI']
    rows = []
    results = []

    baseline_algorithms = ['NoGKernel', 'WLKernel']
    baseline_first_columns = ['\\cite{Schulz2019OnTN}', '\\cite{DBLP:journals/jmlr/ShervashidzeSLMB11}']
    for i, algorithm in enumerate(baseline_algorithms):
        row_string, row_results = baseline_results(algorithm, datasets, 'Reproduce_RuleGNN/Results/Sota/Baseline/', first_column=baseline_first_columns[i], sota=True)
        rows.append(row_string)
        results.append(row_results)

    row, row_results = share_gnn_results('\\MyGNN (ours)', datasets, 'Reproduce_RuleGNN/Results/Sota/', sota=True)
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Random (ours)', datasets, 'Reproduce_RuleGNN/Results/Sota/Random/', sota=True)
    rows.append(row)
    results.append(row_results)
    first_columns = ([f'{x} {baseline_first_columns[i]}' for i, x in enumerate(baseline_algorithms)]
                     + ['\\textbf{\\MyGNN}', '\\textbf{\\MyGNN-Random}'])
    print_table(first_columns, ['\\textbf{NCI1}', '\\textbf{NCI109}', '\\textbf{IMDB-B}', '\\textbf{IMDB-M}']
                , np.array(results),
                [], with_colors=False)


def synthetic_table():
    datasets = ['CSL', 'EvenOddRings2_16', 'EvenOddRingsCount16', 'LongRings100', 'Snowflakes']
    rows = []
    results = []

    baseline_algorithms = ['NoGKernel', 'WLKernel']
    baseline_first_columns = ['\\cite{Schulz2019OnTN}', '\\cite{DBLP:journals/jmlr/ShervashidzeSLMB11}']
    for i, algorithm in enumerate(baseline_algorithms):
        row_string, row_results = baseline_results(algorithm, datasets, 'Reproduce_RuleGNN/Results/Synthetic/Baseline/', first_column=baseline_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    fair_algorithms = ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'GATv2']
    fair_first_columns = ['\\cite{DBLP:conf/iclr/KipfW17}', '\\cite{Hamilton2017InductiveRL}', '\\cite{DBLP:conf/iclr/XuHLJ19}', '\\cite{Velickovic2017GraphAN}', '\\cite{DBLP:conf/iclr/Brody0Y22}']
    fair_path = 'Reproduce_RuleGNN/RESULTS/'
    for i, algorithm in enumerate(fair_algorithms):
        row_string, row_results = fair_gnn_results(algorithm, datasets, fair_path, first_column=fair_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    row, row_results = share_gnn_results('\\MyGNN', datasets, 'Reproduce_RuleGNN/Results/Synthetic/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Random', datasets, 'Reproduce_RuleGNN/Results/Synthetic/Random/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Encoder', datasets, 'Reproduce_RuleGNN/Results/Synthetic/Encoder/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Decoder', datasets, 'Reproduce_RuleGNN/Results/Synthetic/Decoder/')
    rows.append(row)
    results.append(row_results)

    first_columns = ([f'{x} {baseline_first_columns[i]}' for i, x in enumerate(baseline_algorithms)]
                        + [f'{x} {fair_first_columns[i]}' for i, x in enumerate(fair_algorithms)]
                        + ['\\textbf{\\MyGNN (ours)}', '\\textbf{\\MyGNN-Random (ours)}', '\\textbf{\\MyGNN-Encoder (ours)}', '\\textbf{\\MyGNN-Decoder (ours)}'])
    print_table(first_columns, ['\\textbf{CSL}', '\\textbf{EvenOddRings2}', '\\textbf{EvenOddRingsCount}', '\\textbf{LongRings}', '\\textbf{Snowflakes}']
                , np.array(results),
                [2, 7])


def features_evaluation():
    fair_algorithms = ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'GATv2']
    fair_first_columns = ['\\cite{DBLP:conf/iclr/KipfW17}', '\\cite{Hamilton2017InductiveRL}',
                          '\\cite{DBLP:conf/iclr/XuHLJ19}', '\\cite{Velickovic2017GraphAN}',
                          '\\cite{DBLP:conf/iclr/Brody0Y22}']

    datasets = ['NCI1', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI']
    datasets_features = ['NCI1Features', 'DHFRFeatures', 'IMDB-BINARYFeatures', 'IMDB-MULTIFeatures']
    rows = []
    features_results = []
    results = []
    fair_path = 'Reproduce_RuleGNN/RESULTS/'

    for i, algorithm in enumerate(fair_algorithms):
        row_string, row_results = fair_gnn_results(algorithm, datasets, fair_path,
                                                   first_column=fair_first_columns[i])
        rows.append(row_string)
        results.append(row_results)

    for i, algorithm in enumerate(fair_algorithms):
        row_string, row_results = fair_gnn_results(algorithm, datasets_features, fair_path,
                                                   first_column=fair_first_columns[i])
        rows.append(row_string)
        features_results.append(row_results)

    results = np.array(results)
    features_results = np.array(features_results)
    results = features_results - results
    # round the results to 1 decimal place
    results = np.round(results, 1)

    first_columns = [f'{x} {fair_first_columns[i]}' for i, x in enumerate(fair_algorithms)]
    print_table(first_columns, ['\\textbf{NCI1}', '\\textbf{DHFR}', '\\textbf{IMDB-B}', '\\textbf{IMDB-M}']
                , np.array(results),
                [], with_colors=False, with_std=False, positive_negative_colors=True)


def ablation_threshold(dataset='NCI1'):
    ablation_results = dict()
    for i in list(range(1, 21)) + [30, 40, 50]:
        path = f'Reproduce_RuleGNN/Results/Ablation/{i}/'
        # check if the path exists
        if Path(path).exists():
            results_str, results = share_gnn_results('ShareGNN', [dataset], path, False)
            if i in ablation_results and isinstance(ablation_results[i], dict):
                ablation_results[i]['accuracy'] = results[0][0]
                ablation_results[i]['std'] = results[0][1]
            else:
                ablation_results[i] = {'accuracy': results[0][0], 'std': results[0][1]}
    # get number of parameters
    for i in list(range(1, 21)) + [30, 40, 50]:
        path = f'Reproduce_RuleGNN/Results/Ablation/{i}/'
        if Path(path).exists():
            # get the file from results folder that contains Best and Network
            for file in Path(f'{path}/{dataset}/Results').iterdir():
                if 'Best_Configuration' in file.name and 'Network' in file.name:
                    with open(file, 'r') as f:
                        data = f.read()
                        data = data.split('\n')
                        for line in data:
                            if 'Total trainable parameters' in line:
                                num_parameters = int(line.split(':')[-1].strip())
                                ablation_results[i]['parameters'] = num_parameters
                                break
                    break
    # get avg best epoch
    for i in list(range(1, 21)) + [30, 40, 50]:
        path = f'Reproduce_RuleGNN/Results/Ablation/{i}/'
        if Path(path).exists():
            # get the file from results folder that contains Best and Network
            df = pd.read_csv(f'{path}/{dataset}/summary_best_mean.csv', delimiter=",")
            ablation_results[i]['mean_epoch'] = df['Epoch Mean'].values[0]
    # get avg best epoch and avg epoch runtime
    for i in list(range(1, 21)) + [30, 40, 50]:
        path = f'Reproduce_RuleGNN/Results/Ablation/{i}/'
        if Path(path).exists():
            # get the file from results folder that contains Best and Network
            df_all = None
            for file in Path(f'{path}/{dataset}/Results').iterdir():
                if f'{dataset}_Configuration' in file.name and '.csv' in file.suffix:
                    df = pd.read_csv(file, delimiter=";")
                    # concatenate the dataframes
                    if df_all is None:
                        df_all = df
                    else:
                        df_all = pd.concat([df_all, df], ignore_index=True)
            # get the mean of all epoch times
            mean_epoch_time = df_all['EpochTime'].mean()
            ablation_results[i]['mean_epoch_time'] = mean_epoch_time

    fig, ax1 = plt.subplots()

    # ticks inside
    plt.tick_params(axis='both', direction='in')
    # set title to dataset
    plt.title(f'{dataset}')
    # create a bar plot with x-axis as keys of ablation_results and y-axis as accuracy
    ax1.errorbar(ablation_results.keys(), [ablation_results[i]['accuracy'] for i in ablation_results],
                 yerr=[ablation_results[i]['std'] for i in ablation_results], fmt='o', capsize=5)
    ax1.set_ylabel('Accuracy in \\%')
    # set range to 80 - 90
    #ax1.set_ylim([80, 90])


    # add number of parameters to the right y-axis in thousand
    ax2 = plt.gca().twinx()
    # ticks at the inside
    ax2.tick_params(axis='y', direction='in')
    ax2.plot(ablation_results.keys(), [ablation_results[i]['parameters']/1000 for i in ablation_results], 'r', marker='s')
    ax2.set_ylabel('Parameters (in thousands)')
    # set range to 0 - 400
    #ax2.set_ylim([0, 400])

    #  add one legend for both axes
    plt.figlegend(['Accuracy in \\%', 'Parameters (in thousands)'], loc=(0.175, 0.85), ncols=2)
    # set ticks to list(range(1, 21)) + [30, 40, 50]
    plt.xticks(list(range(1, 21, 2)))
    # set x-axis label to the figure
    ax1.set_xlabel('Minimum \\# of Occurrences per Shared Weight (Encoder)')
    plt.savefig(f'Reproduce_RuleGNN/Results/Ablation/ablation_threshold_{dataset}.pdf', bbox_inches='tight', backend='pgf')

    pass





def main():
    ablation_threshold('NCI1')
    ablation_threshold('IMDB-BINARY')
    fair_table_full()
    print('\n\n\n\n')
    sota_baseline_and_share()
    print('\n\n\n\n')
    features_evaluation()
    #synthetic_table()


if __name__ == '__main__':
    main()