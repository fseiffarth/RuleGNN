from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

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
        lamarr_colors = [aqua, white, fuchsia]  # Color 3 (RGB values)

        positions = [0.0, 0.5, 1.0]  # Positions of the colors (range: 0.0 to 1.0)

        # Create a colormap using LinearSegmentedColormap
        self.cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, lamarr_colors)))

class GraphDrawing:
    def __init__(self, node_size=10.0, edge_width=1.0, weight_edge_width=1.0, weight_arrow_size=5.0, edge_color='black', edge_alpha=1, node_color='black', draw_type=None, colormap=plt.get_cmap('tab20')):
        self.node_size = node_size
        self.edge_width = edge_width
        self.weight_edge_width = weight_edge_width
        self.edge_color = edge_color
        self.edge_alpha = edge_alpha
        self.node_color = node_color
        self.arrow_size = weight_arrow_size
        self.draw_type = draw_type
        self.colormap = colormap