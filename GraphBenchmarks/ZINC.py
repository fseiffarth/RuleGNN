# download ZINC data a subset of ZINC dataset
from torch_geometric.datasets import ZINC

from GraphData.GraphData import zinc_to_graph_data


def main(pytorch_geometric=None):
    zinc_train = ZINC(root="../GraphData/ZINC/", subset=True, split='train')
    zinc_val = ZINC(root="../GraphData/ZINC/", subset=True, split='val')
    zinc_test = ZINC(root="../GraphData/ZINC/", subset=True, split='test')
    # zinc to networkx
    networkx_graphs = zinc_to_graph_data(zinc_train, zinc_val, zinc_test, "ZINC")
    pass
    
    
if __name__ == "__main__":
    main()