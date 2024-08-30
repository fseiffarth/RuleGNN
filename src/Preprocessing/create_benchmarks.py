from typing import List

import networkx as nx
import numpy as np

from src.utils.snowflake_generation import Snowflakes
from src.Preprocessing.create_splits import create_splits
from src.utils.save_distances import save_distances
from src.Preprocessing.create_labels import save_standard_labels, save_cycle_labels, save_subgraph_labels
from src.utils.utils import save_graphs




# create function description


def main(output_path="BenchmarkGraphs/", benchmarks=None):
    for name in benchmarks:
        if name == "EvenOddRings1_16":
            graphs, labels = even_odd_rings(data_size=1200, ring_size=16, difficulty=1, count=False)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../../Data/Distances/")
            save_standard_labels(output_path, [name], label_path="../../Data/Labels/")
            create_splits(name, data_path=output_path, output_path="../../Data/Splits/")
        if name == "EvenOddRings2_16":
            graphs, labels = even_odd_rings(data_size=1200, ring_size=16, difficulty=2, count=False)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../../Data/Distances/")
            save_standard_labels(output_path, [name], label_path="../../Data/Labels/")
            create_splits(name, data_path=output_path, output_path="../../Data/Splits/")
        if name == "EvenOddRings2_100":
            graphs, labels = even_odd_rings(data_size=1200, ring_size=100, difficulty=2, count=False)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../../Data/Distances/")
            save_standard_labels(output_path, [name], label_path="../../Data/Labels/")
            create_splits(name, data_path=output_path, output_path="../../Data/Splits/")
        if name == "EvenOddRings3_16":
            graphs, labels = even_odd_rings(data_size=1200, ring_size=16, difficulty=3, count=False)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../../Data/Distances/")
            save_standard_labels(output_path, [name], label_path="../../Data/Labels/")
            create_splits(name, data_path=output_path, output_path="../../Data/Splits/")
        if name == "EvenOddRingsCount16":
            graphs, labels = even_odd_rings(data_size=1200, ring_size=16, count=True)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../../Data/Distances/")
            save_standard_labels(output_path, [name], label_path="../../Data/Labels/")
            create_splits(name, data_path=output_path, output_path="../../Data/Splits/")
        if name == "EvenOddRingsCount100":
            graphs, labels = even_odd_rings(data_size=1200, ring_size=100, count=True)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../../Data/Distances/")
            save_standard_labels(output_path, [name], label_path="../../Data/Labels/")
            create_splits(name, data_path=output_path, output_path="../../Data/Splits/")
        if name == "LongRings100":
            graphs, labels = long_rings(data_size=1200, ring_size=100)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../../Data/Distances/")
            save_standard_labels(output_path, [name], label_path="../../Data/Labels/")
            create_splits(name, data_path=output_path, output_path="../../Data/Splits/")
        if name == "LongRings8":
            graphs, labels = long_rings(data_size=1200, ring_size=8)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../../Data/Distances/")
            save_standard_labels(output_path, [name], label_path="../../Data/Labels/")
            create_splits(name, data_path=output_path, output_path="../../Data/Splits/")
        if name == "LongRings16":
            graphs, labels = long_rings(data_size=1200, ring_size=16)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../../Data/Distances/")
            save_standard_labels(output_path, [name], label_path="../../Data/Labels/")
            create_splits(name, data_path=output_path, output_path="../../Data/Splits/")
        if name == "SnowflakesCount":
            graphs, labels = Snowflakes(smallest_snowflake=3, largest_snowflake=6, flakes_per_size=200, seed=764, generation_type="count")
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../../Data/Distances/")
            save_standard_labels(output_path, [name], label_path="../../Data/Labels/")
            save_cycle_labels(output_path, [name], length_bound=3, cycle_type="induced", label_path="../../Data/Labels/")
            save_cycle_labels(output_path, [name], length_bound=6, cycle_type="induced", label_path="../../Data/Labels/")
            save_subgraph_labels(output_path, [name], subgraphs=[nx.cycle_graph(5)], id=0, label_path="../../Data/Labels/")
            save_subgraph_labels(output_path, [name], subgraphs=[nx.cycle_graph(4)], id=1, label_path="../../Data/Labels/")
            save_subgraph_labels(output_path, [name], subgraphs=[nx.cycle_graph(4), nx.cycle_graph(5)], id=2, label_path="../../Data/Labels/")
            create_splits(name, data_path=output_path, output_path="../../Data/Splits/")
        if name == "Snowflakes":
            graphs, labels = Snowflakes(smallest_snowflake=3, largest_snowflake=12, flakes_per_size=100, seed=764, generation_type="binary")
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../../Data/Distances/")
            save_standard_labels(output_path, [name], label_path="../../Data/Labels/")
            #save_circle_labels(output_path, [name], length_bound=3, cycle_type="induced", label_path="../BenchmarkGraphs/Labels/")
            #save_circle_labels(output_path, [name], length_bound=5, cycle_type="induced", label_path="../BenchmarkGraphs/Labels/")
            #save_circle_labels(output_path, [name], length_bound=6, cycle_type="induced", label_path="../BenchmarkGraphs/Labels/")
            #save_subgraph_labels(output_path, [name], subgraphs=[nx.cycle_graph(5)], id=0, label_path="../BenchmarkGraphs/Labels/")
            #save_subgraph_labels(output_path, [name], subgraphs=[nx.cycle_graph(4)], id=1, label_path="../BenchmarkGraphs/Labels/")
            save_subgraph_labels(output_path, [name], subgraphs=[nx.cycle_graph(4), nx.cycle_graph(5)], id=2, label_path="../../Data/Labels/")
            create_splits(name, data_path=output_path, output_path="../../Data/Splits/")
        if name == "CSL_original":
            from src.Preprocessing.csl import CSL
            csl = CSL()
            graph_data = csl.get_graphs(with_distances=False)
            save_graphs(output_path, name, graph_data.graphs, graph_data.graph_labels)
if __name__ == "__main__":
    #main(benchmarks=["EvenOddRings1_16", "EvenOddRings2_16", "EvenOddRings3_16"])
    #main(benchmarks=["LongRings16"])
    #main(benchmarks=["LongRings100"])
    #main(benchmarks=["EvenOddRingsCount16"])
    main(benchmarks=["EvenOddRings2_16"])
    #main(benchmarks=["LongRings100", "EvenOddRings2_16", "EvenOddRingsCount16", "Snowflakes"])
    # main(benchmarks=["EvenOddRings2_16", "EvenOddRings2_120"])
    # main(benchmarks=["EvenOddRings3_16", "EvenOddRings3_120"])

    # main(benchmarks=["LongRingsLabeled16", "LongRings100", "LongRings8"])
