# define a class that loads the nel format data and convert it to pyg format
import torch
from torch_geometric.data import InMemoryDataset


class NELInMemory(InMemoryDataset):
    def __init__(self, db_name, root, transform=None, pre_transform=None):
        super(NELInMemory, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.db_name}_Labels.txt', f'{self.db_name}_Nodes.txt', f'{self.db_name}_Edges.txt']

    @property
    def processed_file_names(self):
        return [f'{self.db_name}.pt']

    def process(self):
        data_list = []
        # process the data here
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def test():
    db_name = 'Snowflakes'
    root = 'Reproduce_RuleGNN/Data/SyntheticDatasets/Snowflakes'
    dataset = NELInMemory(db_name, root)

if __name__ == '__main__':
    test()