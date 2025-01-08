import json

import numpy as np
import pandas as pd


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
        results.append((test_acc, test_std))
        test_std = round(test_std, 1)
        out_str += f' & ${test_acc} \\pm {test_std}$'
    return out_str, results

def print_table(first_columns, results):

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
                if accuracies[i, j] in best_column_values[:, j]:
                    # get index of the value
                    index = np.where(best_column_values[:, j] == accuracies[i, j])[0][0]
                    if index == 0:
                        line += f' & \\ThirdColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                    elif index == 1:
                        line += f' & \\SecondColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                    elif index == 2:
                        line += f' & \\FirstColor{{{accuracies[i, j]} \\pm {stds[i, j]}}}'
                else:
                    line += f' & {accuracies[i, j]} \\pm {stds[i, j]}'
            lines.append(line)
    # print the lines
    for line in lines:
        print(line)

def fair_table():
    # print the large table
    datasets = ['NCI1', 'NCI109', 'Mutagenicity', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI']
    rows = []
    results = []
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

    first_columns = fair_first_columns + ['\\textbf{\\MyGNN (ours)}', '\\textbf{\\MyGNN-Random (ours)}', '\\textbf{\\MyGNN-Encoder (ours)}', '\\textbf{\\MyGNN-Decoder (ours)}']

    print_table(first_columns, results)



def main():
    fair_table()


if __name__ == '__main__':
    main()