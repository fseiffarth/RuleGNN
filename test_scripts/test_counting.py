from test_scripts.counting_from_paper import GraphCount


def main():
    # get relative path to the tmp directory
    root = "/home/florian/Documents/Code/GNNs/RuleGNN/test_scripts/"

    graph_count = GraphCount(root=root, split="train", task="triangle")
    pass

if __name__ == '__main__':
    main()