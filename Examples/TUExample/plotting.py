from pathlib import Path

from scripts.ExperimentMain import ExperimentMain
from scripts.WeightVisualization import WeightVisualization, GraphDrawing

def main():
    experiment = ExperimentMain(Path('Examples/TUExample/Configs/config_main.yml'))

    db_name = 'DHFR'
    graph_ids = [0, 1, 2]
    filter_sizes = (None, 10, 3)
    graph_drawing = (
        GraphDrawing(node_size=40, edge_width=1),
        GraphDrawing(node_size=40, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10,
                     colormap=CustomColorMap().cmap)
    )
    ww = WeightVisualization(db_name=db_name, experiment=experiment)
    for validation_id in range(10):
        for run in range(3):
            ww.visualize(graph_ids, run=run, validation_id=validation_id, graph_drawing=graph_drawing,
                         filter_sizes=filter_sizes)


if __name__ == '__main__':
    main()