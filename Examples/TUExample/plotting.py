import matplotlib.colors as mcolors

from scripts.WeightVisualization import WeightVisualization, GraphDrawing


class CustomColorMap:
    def __init__(self):
        aqua = (0.0, 0.6196, 0.8902)
        # 89,189,247
        skyblue = (0.3490, 0.7412, 0.9686)
        fuchsia = (232 / 255.0, 46 / 255.0, 130 / 255.0)
        violet = (152 / 255.0, 48 / 255.0, 130 / 255.0)
        white = (1.0, 1.0, 1.0)
        # darknavy 12,18,43
        darknavy = (12 / 255.0, 18 / 255.0, 43 / 255.0)

        # Define the three colors and their positions
        lamarr_colors = [aqua, white, fuchsia] # Color 3 (RGB values)

        positions = [0.0, 0.5, 1.0]  # Positions of the colors (range: 0.0 to 1.0)

        # Create a colormap using LinearSegmentedColormap
        self.cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, lamarr_colors)))

def main():
    db_name = 'PTC_FM'
    main_config_file = 'Examples/TUExample/Configs/config_main.yml'
    experiment_config_file = 'Examples/TUExample/Configs/config_experiment.yml'
    graph_ids = [0, 1, 2]
    filter_sizes = (None, 10, 3)
    graph_drawing = (
        GraphDrawing(node_size=40, edge_width=1),
        GraphDrawing(node_size=40, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10,
                     colormap=CustomColorMap().cmap)
    )
    ww = WeightVisualization(db_name=db_name, main_config=main_config_file, experiment_config=experiment_config_file)
    ww.visualize(graph_ids, run=0, validation_id=0, graph_drawing=graph_drawing, filter_sizes=filter_sizes)

if __name__ == '__main__':
    main()