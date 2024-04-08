import os

import numpy as np
import torch
import torch_geometric.datasets
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from GraphData.DataSplits.load_splits import Load_Splits


class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        # linear output layer
        self.linear = torch.nn.Linear(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # linear layer for output
        # take the mean of the node embeddings
        x = global_mean_pool(x, data.batch)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)


class GNN(torch.nn.Module):
    def __init__(self, db_name, model, run_num, validation_num, training_data, validate_data, test_data, seed):
        super().__init__()
        self.db_name = db_name
        self.model = model
        self.dataset = self.load_dataset()

        self.run_num = run_num
        self.validation_num = validation_num
        self.training_data = training_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.seed = seed

    def load_dataset(self):
        # load the data using TUDataset from PyTorch Geometric
        # create Datasets folder if it does not exist
        if not os.path.exists('Datasets'):
            os.makedirs('Datasets')
        return torch_geometric.datasets.TUDataset(root='Datasets', name=self.db_name)
    def Run(self):
        # get training, validation and test data
        training_data = self.dataset[self.training_data]
        validate_data = self.dataset[self.validate_data]
        test_data = self.dataset[self.test_data]

        training_labels = training_data.y
        validate_labels = validate_data.y
        test_labels = test_data.y

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN(self.dataset).to(device)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        model.train()
        for epoch in range(200):
            loader = torch_geometric.loader.DataLoader(training_data, batch_size=16, shuffle=True)
            for batch in loader:
                optimizer.zero_grad()
                out = model(batch)
                loss = F.nll_loss(out, batch.y)
                loss.backward()
                # get epoch accuracy
                pred = out.argmax(dim=1)
                acc = accuracy_score(batch.y.cpu().numpy(), pred.cpu().numpy())
                optimizer.step()

                # get the validation accuracy
                model.eval()
                loader = torch_geometric.loader.DataLoader(validate_data, batch_size=self.validate_data.size, shuffle=False)
                for i,batch in enumerate(loader):
                    out = model(batch)
                    pred = out.argmax(dim=1)
                    acc = accuracy_score(validate_labels, pred)
                    # print epoch, batch loss and validation accuracy
                    print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}, Validation Accuracy: {acc}')
                    print(f'Epoch: {epoch}, Loss: {loss.item()}, Validation Accuracy: {acc}')





def main(db_name):
    validation_size = 10
    if db_name == "CSL":
        validation_size = 5
    for validation_id in range(validation_size):
        # three runs
        for i in range(3):
            data = Load_Splits("../../GraphData/DataSplits", db_name)
            test_data = np.asarray(data[0][validation_id], dtype=int)
            training_data = np.asarray(data[1][validation_id], dtype=int)
            validate_data = np.asarray(data[2][validation_id], dtype=int)

            gcn = GNN(db_name, model=GCN, run_num=i, validation_num=validation_id, training_data=training_data,
                      validate_data=validate_data, test_data=test_data, seed=i)

            gcn.Run()


if __name__ == "__main__":
    main("NCI1")
