from enum import Enum


class LayerTypes(Enum):
    """All currently supported layer types."""
    INVARIANT_BASED_CONVOLUTION = 'invariant_based_convolution'
    INVARIANT_BASED_AGGREGATION = 'invariant_based_aggregation'
    LINEAR = 'linear'
    RESHAPE = 'reshape'
    LAYER_NORM = 'layer_norm'
    GLOBAL_POOLING = 'global_pooling'
    ACTIVATION = 'activation'
    DROPOUT = 'dropout'
    BATCH_NORM = 'batch_norm'

    ### GNN specific layers
    GCN_CONVOLUTION = 'gcn_convolution'
    GAT_CONVOLUTION = 'gat_convolution'
    GATv2_CONVOLUTION = 'gatv2_convolution'
    GIN_CONVOLUTION = 'gin_convolution'
    SAGE_CONVOLUTION = 'sage_convolution'
