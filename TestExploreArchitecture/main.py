from pathlib import Path

import yaml
from ogb.graphproppred import PygGraphPropPredDataset


def main():
    PygGraphPropPredDataset(name="ogbg-molfreesolv", root='dataset/')
    #network_architecture = yaml.safe_load(open('ReproduceExtended/configs/network_zinc_multihead_best.yml'))
    # get all networks
    #networks = network_architecture['networks']
    pass

if __name__ == '__main__':
    main()