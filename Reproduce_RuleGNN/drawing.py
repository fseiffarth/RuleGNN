import matplotlib

from scripts.WeightVisualization import WeightVisualization, GraphDrawing

class CustomColorMap:
    def __init__(self):
        # rgb colors
        rgb1 = (0,158,227)
        rgb2 = (115,115,97)
        rgb3 = (232,46,130)
        # normalize to [0,1]
        rgb1 = tuple([x / 255 for x in rgb1])
        rgb2 = tuple([x / 255 for x in rgb2])
        rgb3 = tuple([x / 255 for x in rgb3])
        # create linear segmented colormap between the colors
        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', [rgb1, rgb3])

def main():
    main_config_file = 'Reproduce_RuleGNN/Configs/main_config.yml'

    ### Snowflakes
    db_name = 'Snowflakes'
    experiment_config_file = 'Reproduce_RuleGNN/Configs/config_Snowflakes.yml'
    graph_ids = [0, 300, 900]
    graph_drawing = (
        GraphDrawing(node_size=10, edge_width=0.5, draw_type='kawai', colormap=CustomColorMap().cmap),
        GraphDrawing(node_size=5, edge_width=3, edge_arrow_size=5, colormap=CustomColorMap().cmap)
    )
    ww = WeightVisualization(db_name=db_name, main_config=main_config_file, experiment_config=experiment_config_file)
    ww.visualize(graph_ids, run=0, validation_id=0, graph_drawing=graph_drawing, filter_sizes=(None,))
    ### DHFR
    db_name = 'DHFR'
    experiment_config_file = 'Reproduce_RuleGNN/Configs/config_DHFR.yml'
    graph_ids = [0, 40, 80]
    filter_sizes = (None, 10, 3)
    graph_drawing = (
        GraphDrawing(node_size=20, edge_width=1),
        GraphDrawing(node_size=10, edge_width=3, edge_arrow_size=8, colormap=CustomColorMap().cmap)
    )
    ww = WeightVisualization(db_name=db_name, main_config=main_config_file, experiment_config=experiment_config_file)
    ww.visualize(graph_ids, run=0, validation_id=0, graph_drawing=graph_drawing, filter_sizes=filter_sizes)


if __name__ == '__main__':
    main()