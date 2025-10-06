# Custom Layers

Each layer is represented by a dict in the network configuration file, e.g.:

> ```yml
> networks:
> - - { layer_type: my_layer,
>     out_channels: 32,
>     bias: True,
>     activation: relu }
> ```

If you want to implement a custom layer, you need to go through the following steps:

1. **Implement the Layer Class**: Create a new Python class for your layer that inherits from `FrameworkLayers`. Implement the `__init__` and `forward` methods.

   ```python
   import torch
   import torch.nn as nn
   from src.Architectures.Layers import FrameworkLayers

   class MyLayer(FrameworkLayers):
       def __init__(self, layer_dict):
           super(MyLayer, self).__init__()
           self.out_channels = layer_dict.get('out_channels', 32)
           self.bias = layer_dict.get('bias', True)
           self.activation = getattr(nn, layer_dict.get('activation', 'ReLU'))() if layer_dict.get('activation') else None
           # Define other layer components here

       def forward(self, x):
           # Implement the forward pass
           if self.activation:
               x = self.activation(x)
           return x
   ```
   
2. **Register the Layer**: Add your new layer class to the `layer_type_to_class` dictionary in `src/Architectures/Layers/__init__.py`.

   ```python