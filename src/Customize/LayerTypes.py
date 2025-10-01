from enum import Enum


class LayerTypes(Enum):
    """All currently supported layer types."""
    INVARIANT_BASED_CONVOLUTION = 'invariant_based_convolution'
    INVARIANT_BASED_AGGREGATION = 'invariant_based_aggregation'
    LINEAR = 'linear'
    RESHAPE = 'reshape'
    LAYER_NORM = 'layer_norm'
    GCN_CONVOLUTION = 'gcn_convolution'
    MEAN_AGGREGATION = 'mean_aggregation'