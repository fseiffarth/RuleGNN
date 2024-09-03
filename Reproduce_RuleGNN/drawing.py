import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

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
        lamarr_colors = [darknavy, white, fuchsia] # Color 3 (RGB values)

        positions = [0.0, 0.5, 1.0]  # Positions of the colors (range: 0.0 to 1.0)

        # Create a colormap using LinearSegmentedColormap
        self.cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, lamarr_colors)))
def main():
    main_config_file = 'Reproduce_RuleGNN/Configs/main_config.yml'
    ### DHFR
    db_name = 'DHFR'
    experiment_config_file = 'Reproduce_RuleGNN/Configs/config_DHFR.yml'
    graph_ids = [0,1,2]
    filter_sizes = (None, 10, 3)
    graph_drawing = (
        GraphDrawing(node_size=40, edge_width=1),
        GraphDrawing(node_size=40, edge_width=2, edge_arrow_size=8, colormap=CustomColorMap().cmap)
    )
    ww = WeightVisualization(db_name=db_name, main_config=main_config_file, experiment_config=experiment_config_file)
    for run in range(3):
        ww.visualize(graph_ids, run=run, validation_id=3, graph_drawing=graph_drawing, filter_sizes=filter_sizes)

    ### Snowflakes
    db_name = 'Snowflakes'
    experiment_config_file = 'Reproduce_RuleGNN/Configs/config_Snowflakes.yml'
    graph_ids = [0, 40, 80]
    graph_drawing = (
        GraphDrawing(node_size=20, edge_width=1, draw_type='kawai'),
        GraphDrawing(node_size=2, edge_width=1, edge_arrow_size=5, colormap=CustomColorMap().cmap)
    )
    ww = WeightVisualization(db_name=db_name, main_config=main_config_file, experiment_config=experiment_config_file)
    for run in range(3):
        ww.visualize(graph_ids, run=run, validation_id=0, graph_drawing=graph_drawing, filter_sizes=(None,))





    ### NCI1

if __name__ == '__main__':
    main()