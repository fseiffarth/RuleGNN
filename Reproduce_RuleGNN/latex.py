import json
from pathlib import Path

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
        with open(f"{path}/{algorithm}_{dataset}_assessment/10_NESTED_CV/assessment_results.json", 'r') as f:
            data = json.load(f)
            pass
        # open summary_best_mean.csv
        test_acc = data['avg_TS_score']
        test_std = data['std_TS_score']
        # round the test accuracy and test std to 2 decimal places
        test_acc = round(test_acc, 1)
        test_std = round(test_std, 1)
        results.append((test_acc, test_std))
        out_str += f' & ${test_acc} \\pm {test_std}$'
    return out_str, results

def print_table(first_columns, column_names, results, rules:list[int], with_colors:bool=True):

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
                        line += f' & \\ThirdColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                    elif index == 1:
                        line += f' & \\SecondColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                    elif index == 2:
                        line += f' & \\FirstColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                else:
                    line += f' & ${accuracies[i, j]} \\pm {stds[i, j]}$'
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
                     + ['\\textbf{\\MyGNN (ours)}', '\\textbf{\\MyGNN-Random (ours)}'])
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

    row, row_results = share_gnn_results('\\MyGNN (ours)', datasets, 'Reproduce_RuleGNN/Results/Synthetic/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Random (ours)', datasets, 'Reproduce_RuleGNN/Results/Synthetic/Random/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Encoder (ours)', datasets, 'Reproduce_RuleGNN/Results/Synthetic/Encoder/')
    rows.append(row)
    results.append(row_results)
    row, row_results = share_gnn_results('\\MyGNN-Decoder (ours)', datasets, 'Reproduce_RuleGNN/Results/Synthetic/Decoder/')
    rows.append(row)
    results.append(row_results)

    first_columns = ([f'{x} {baseline_first_columns[i]}' for i, x in enumerate(baseline_algorithms)]
                        + [f'{x} {fair_first_columns[i]}' for i, x in enumerate(fair_algorithms)]
                        + ['\\textbf{\\MyGNN (ours)}', '\\textbf{\\MyGNN-Random (ours)}', '\\textbf{\\MyGNN-Encoder (ours)}', '\\textbf{\\MyGNN-Decoder (ours)}'])
    print_table(first_columns, ['\\textbf{CSL}', '\\textbf{EvenOddRings2}', '\\textbf{EvenOddRingsCount}', '\\textbf{LongRings}', '\\textbf{Snowflakes}']
                , np.array(results),
                [2, 7])


def main():
    fair_table()
    sota_baseline_and_share()
    synthetic_table()


if __name__ == '__main__':
    main()