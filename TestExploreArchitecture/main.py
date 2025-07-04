from pathlib import Path

import yaml


def main():
    network_architecture = yaml.safe_load(open('ReproduceExtended/configs/network_zinc_multihead_best.yml'))
    # get all networks
    networks = network_architecture['networks']
    pass

if __name__ == '__main__':
    main()