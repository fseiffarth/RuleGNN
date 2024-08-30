# convert graphs from TU_Dortmund benchmark graph dataset to NEL format
from pathlib import Path

import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.remote_backend_utils import num_nodes
from torch_geometric.datasets import TUDataset

from src.utils.utils import save_graphs


def tu_to_nel(db_name: str, out_path: Path = Path('Data/TUDatasets')):
    '''
    Convert graphs from TU Dortmund benchmark graph dataset to NEL format
    :param db_name: name of the dataset
    :param out_path: path to save the converted graphs
    '''

    # check if the dataset in out_path already exists, i.e., in the raw folder
    if Path(out_path / db_name / 'raw').exists() and len(list(Path(out_path / db_name / 'raw').iterdir())) > 0:
        print(f"Dataset {db_name} already exists in {out_path} . Skip the data generation.")
        return
    # download the dataset
    #create a tmp folder to store the dataset
    if not Path('tmp').exists():
        Path('tmp').mkdir()
    tu_dataset = TUDataset(root=Path('tmp/'), name=db_name, use_node_attr=True, use_edge_attr=True)
    # get the graphs and labels
    labels = []
    # get single graphs from tu_dataset
    graphs = []
    for g in tu_dataset:
        if g.x is not None:
            # analyze g.x which is float attribute and which is one_hot encoding
            # get the sum over abs values
            x_sum = sum(abs(g['x']))
            # get the indices of x which are integers and those which are float True, False
            x_indices = torch.eq(x_sum, x_sum.int())
            # True means integer, False means float
            int_indices = torch.nonzero(x_indices).squeeze()
            float_indices = torch.nonzero(~x_indices).squeeze()

            features = g['x'][:, float_indices]
            one_hot_encoding = g['x'][:, int_indices]
            one_hot_label = torch.argmax(one_hot_encoding, dim=1)
            # add the one hot labels as the first column of the features
            g['x'] = torch.cat((one_hot_label.unsqueeze(1), features), dim=1)

        # get the node and edge attributes
        if g.edge_attr is not None:
            # if edge_attr is one_hot encoding, convert it to integer
            if int(sum(sum(g['edge_attr'])).item()) == g['edge_attr'].shape[0]:
                g['edge_attr'] = torch.argmax(g['edge_attr'], dim=1)
            if g.x is not None:
                graph = torch_geometric.utils.to_networkx(g, to_undirected=True, node_attrs=['x'], edge_attrs=['edge_attr'])
            else:
                graph = torch_geometric.utils.to_networkx(g, to_undirected=True, edge_attrs=['edge_attr'])
            # rename edge data from edge_attr to label
            for edge in graph.edges(data=True):
                edge[2]['label'] = edge[2]['edge_attr']
                del edge[2]['edge_attr']

        else:
            if g.x is not None:
                graph = torch_geometric.utils.to_networkx(g, to_undirected=True, node_attrs=['x'])
            else:
                graph = torch_geometric.utils.to_networkx(g, to_undirected=True)
        if g.x is not None:
            # rename node data from x to label
            for i in graph.nodes(data=True):
                i[1]['label'] = i[1].pop('x')
        # add the graph to the list
        graphs.append(graph)
        labels.append(g["y"].item())
    # save the graphs and labels
    save_graphs(out_path, db_name, graphs, labels, with_degree=False, format='NEL')


def main():
    # convert graphs from TU Dortmund benchmark graph dataset to NEL format
    tu_to_nel('IMDB-BINARY')


if __name__ == '__main__':
    main()