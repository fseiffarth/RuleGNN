import Layers.GraphLayers as layers
import torch
import torch.nn as nn
import RuleFunctions.Rules as rule
from GraphData import GraphData

from Time.TimeClass import TimeClass


###Net for learning graph class
class GraphNetOriginal(nn.Module):
    def __init__(self, graph_data: GraphData, n_node_features, n_node_labels, n_edge_labels, seed, dropout=0,
                 out_classes=2,
                 print_weights=False, print_layer_init=False, save_weights=False, convolution_grad=True,
                 resize_grad=True):
        super(GraphNetOriginal, self).__init__()
        self.graph_data = graph_data
        self.print_weights = print_weights
        self.out_dim = self.graph_data.num_classes
        self.l1 = layers.GraphConvLayer(layer_id=1, seed=seed + 1, graph_data=self.graph_data,
                                        w_distribution_rule=rule.weight_rule_wf,
                                        bias_distribution_rule=rule.node_label_rule, in_features=n_node_features,
                                        n_node_labels=n_node_labels, n_edge_labels=n_edge_labels, n_kernels=1,
                                        bias=True, print_layer_init=print_layer_init,
                                        save_weights=save_weights).double().requires_grad_(convolution_grad)
        self.l2 = layers.GraphConvLayer(layer_id=2, seed=seed + 2, graph_data=self.graph_data,
                                        w_distribution_rule=rule.weight_rule_wf,
                                        bias_distribution_rule=rule.node_label_rule, in_features=n_node_features,
                                        n_node_labels=n_node_labels, n_edge_labels=n_edge_labels, n_kernels=1,
                                        bias=True, print_layer_init=print_layer_init,
                                        save_weights=save_weights).double().requires_grad_(convolution_grad)
        self.l3 = layers.GraphConvLayer(layer_id=3, seed=seed + 3, graph_data=self.graph_data,
                                        w_distribution_rule=rule.weight_rule_wf,
                                        bias_distribution_rule=rule.node_label_rule, in_features=n_node_features,
                                        n_node_labels=n_node_labels, n_edge_labels=n_edge_labels, n_kernels=1,
                                        bias=True, print_layer_init=print_layer_init,
                                        save_weights=save_weights).double().requires_grad_(convolution_grad)
        self.lr = layers.GraphResizeLayer(layer_id=4, seed=seed + 4, graph_data=self.graph_data,
                                          w_distribution_rule=rule.node_label_rule,
                                          in_features=n_node_features, out_features=self.out_dim,
                                          n_node_labels=n_node_labels,
                                          bias=True, print_layer_init=print_layer_init,
                                          save_weights=save_weights).requires_grad_(resize_grad)
        self.lfc1 = nn.Linear(self.out_dim, self.out_dim, bias=True).double()
        self.lfc2 = nn.Linear(self.out_dim, self.out_dim, bias=True).double()
        self.lfc3 = nn.Linear(self.out_dim, self.out_dim, bias=True).double()
        self.lfc_out = nn.Linear(self.out_dim, out_classes, bias=True).double()
        """
        self.lfc2.weight.requires_grad_(False)
        self.lfc1.weight.requires_grad_(False)
        self.lfc3.weight.requires_grad_(False)
        """
        self.dropout = nn.Dropout(dropout)
        self.af = nn.Tanh()
        self.out_af = nn.Tanh()
        self.epoch = 0
        self.timer = TimeClass()

    def forward(self, x, pos):
        self.timer.measure("forward_wlrule")

        # if self.train(True):
        #     x = self.dropout(x)

        x = self.af(self.l1(x, pos))
        # if self.train(True):
        #    x = self.dropout(x)
        x = self.af(self.l2(x, pos))
        # if self.train(True):
        #    x = self.dropout(x)
        x = self.af(self.l3(x, pos))
        self.timer.measure("forward_wlrule")
        # x = self.af(self.l4(x, pos))
        # x = self.af(self.l5(x, pos))
        # print(x)
        # x = self.af(self.lcycle(x, pos))
        """
        if self.epoch == 10:
            gdtgl.draw_graph_node_labels(self.graph_data[0][pos], torch.reshape(x, (len(self.graph_data[0][pos]), -1)).detach().numpy())
        """
        # x = self.af(self.l4(x, pos))
        """
        if self.epoch == 10:
            gdtgl.draw_graph_node_labels(self.graph_data[0][pos], torch.reshape(x, (len(self.graph_data[0][pos]), -1)).detach().numpy())
        """
        # x = self.af(self.l3(x, pos))
        self.timer.measure("forward_resize")
        x = self.af(self.lr(x, pos))
        # if self.train(True):
        #    x = self.dropout(x)
        self.timer.measure("forward_resize")
        self.timer.measure("forward_fc")
        #x = self.af(self.lfc1(x))
        #x = self.af(self.lfc2(x))
        #x = self.af(self.lfc3(x))
        #x = self.out_af(self.lfc_out(x))
        self.timer.measure("forward_fc")
        return x

    def return_info(self):
        return type(self)


class GraphNet(nn.Module):
    def __init__(self, graph_data: GraphData,
                 para,
                 n_node_features, seed, dropout=0,
                 out_classes=2,
                 print_weights=False, print_layer_init=False, save_weights=False, convolution_grad=True,
                 resize_grad=True):
        super(GraphNet, self).__init__()
        self.graph_data = graph_data
        self.print_weights = print_weights
        self.para = para

        self.net_layers = []
        for i, layer in enumerate(para.layers):
            if i < len(para.layers) - 1:
                self.net_layers.append(
                    layers.GraphConvLayer(layer_id=i, seed=seed + i, parameters=para, graph_data=self.graph_data,
                                          w_distribution_rule=rule.weight_rule_wf_dist,
                                          bias_distribution_rule=rule.node_label_rule, in_features=n_node_features,
                                          node_labels=layer.get_layer_string(), n_kernels=1,
                                          bias=True, print_layer_init=print_layer_init, save_weights=save_weights,
                                          distances=layer.distances).float().requires_grad_(convolution_grad))
            else:
                self.net_layers.append(layers.GraphResizeLayer(layer_id=i, seed=seed + i, parameters=para, graph_data=self.graph_data,
                                                               w_distribution_rule=rule.node_label_rule,
                                                               in_features=n_node_features, out_features=out_classes,
                                                               node_labels=layer.get_layer_string(),
                                                               bias=True, print_layer_init=print_layer_init,
                                                               save_weights=save_weights).float().requires_grad_(
                    resize_grad))

        if 'linear_layers' in para.configs and para.configs['linear_layers'] > 0:
            for i in range(para.configs['linear_layers']):
                self.net_layers.append(nn.Linear(out_classes, out_classes, bias=True).float())

        self.dropout = nn.Dropout(dropout)
        if 'activation' in para.configs and para.configs['activation'] == 'None':
            self.af = nn.Identity()
        elif 'activation' in para.configs and para.configs['activation'] == 'Relu':
            self.af = nn.ReLU()
        elif 'activation' in para.configs and para.configs['activation'] == 'LeakyRelu':
            self.af = nn.LeakyReLU()
        else:
            self.af = nn.Tanh()
        if 'output_activation' in para.configs and para.configs['output_activation'] == 'None':
            self.out_af = nn.Identity()
        elif 'output_activation' in para.configs and para.configs['output_activation'] == 'Relu':
            self.out_af = nn.ReLU()
        elif 'output_activation' in para.configs and para.configs['output_activation'] == 'LeakyRelu':
            self.out_af = nn.LeakyReLU()
        else:
            self.out_af = nn.Tanh()
        self.epoch = 0
        self.timer = TimeClass()

    def forward(self, x, pos):
        for i, layer in enumerate(self.net_layers):
            if i < len(self.net_layers) - 1:
                x = self.af(layer(x, pos))
            else:
                if 'linear_layers' in self.para.configs and self.para.configs['linear_layers'] > 0:
                    x = self.out_af(layer(x))
                else:
                    x = self.out_af(layer(x, pos))
        return x

    def return_info(self):
        return type(self)


class GraphTestNet(nn.Module):
    def __init__(self, graph_data: GraphData, n_node_features, n_node_labels, n_edge_labels, distance_list=[],
                 cycle_list=[],
                 print_weights=False):
        super(GraphTestNet, self).__init__()
        self.graph_data = graph_data
        self.print_weights = print_weights
        self.l1 = layers.GraphConvLayer(graph_data=self.graph_data, w_distribution_rule=rule.weight_rule_wf_dist,
                                        bias_distribution_rule=rule.node_label_rule, in_features=n_node_features,
                                        n_node_labels=n_node_labels, n_edge_labels=n_edge_labels, n_kernels=1,
                                        bias=False, distance_list=distance_list, distances=[0, 1, 2, 3])
        self.l2 = layers.GraphConvLayer(graph_data=self.graph_data, w_distribution_rule=rule.weight_rule_wf_dist,
                                        bias_distribution_rule=rule.node_label_rule, in_features=n_node_features,
                                        n_node_labels=n_node_labels, n_edge_labels=n_edge_labels, n_kernels=1,
                                        bias=False, distance_list=distance_list, distances=[0, 1, 2, 3])

        self.lr = layers.GraphResizeLayer(graph_data=self.graph_data,
                                          w_distribution_rule=rule.node_label_rule,
                                          in_features=n_node_features, out_features=10, n_node_labels=n_node_labels,
                                          bias=False)
        self.lfc1 = nn.Linear(10, 10, bias=True).double()
        self.lfc2 = nn.Linear(10, 10, bias=True).double()
        self.lfc3 = nn.Linear(10, 1, bias=True).double()
        self.dropout = nn.Dropout(0.2)
        self.af = nn.Tanh()
        self.out_af = nn.Sigmoid()
        self.epoch = 0
        self.timer = TimeClass()

    def forward(self, x, pos):
        # x = self.dropout(x)
        # print("After Dropout", x)
        # print(self.l1.name, [x.item() for x in self.l1.Param_W])
        # print(self.l2.name, [x.item() for x in self.l2.Param_W])
        # print(self.lr.name, [x.item() for x in self.lr.Param_W])

        self.timer.measure("forward_wlrule")
        with torch.no_grad():
            x = self.l1(x, pos)
            # print("After First Layer", x)
            x = self.af(x)
            x = self.l2(x, pos)
            # print("After Second Layer", x)
            x = self.af(x)
            # x = self.af(self.l3(x, pos))
            self.timer.measure("forward_wlrule")
            self.timer.measure("forward_resize")
            x = self.lr(x, pos)
        # print("After Resize Layer", x)
        x = self.out_af(self.lfc3(self.af(self.lfc2(self.af(self.lfc1(x))))))
        self.timer.measure("forward_resize")
        return x

    def return_info(self):
        return type(self)
