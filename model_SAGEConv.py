import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, RGCNConv
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, 256)
        self.conv2 = SAGEConv(256, 128)
        self.conv3 = SAGEConv(128, 64)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def forward(self, x_miRNA, x_disease, edge_label_index):
        # Convert node embeddings to edge-level representations:
        edge_feature_RNA = x_miRNA[edge_label_index[0]]
        edge_feature_disease = x_disease[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feature_RNA * edge_feature_disease).sum(dim=-1)

    def prediction(self, x_miRNA, x_disease):
        res = torch.matmul(x_miRNA, x_disease.t())
        return res

class Model(torch.nn.Module):
    def __init__(self, data, miRNA_features, disease_features, hidden_channels):
        super().__init__()
        self.miRNA_lin = torch.nn.Linear(miRNA_features, hidden_channels)
        self.miRNA_emb = torch.nn.Embedding(data["miRNA"].num_nodes, hidden_channels)

        self.disease_lin = torch.nn.Linear(disease_features, hidden_channels)
        self.disease_emb = torch.nn.Embedding(data["disease"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData):
        x_dict = {
            "miRNA": self.miRNA_lin(data["miRNA"].x) + self.miRNA_emb(data["miRNA"].node_id),
            "disease": self.disease_lin(data["disease"].x) + self.disease_emb(data["disease"].node_id)
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["miRNA"],
            x_dict["disease"],
            data["miRNA", "associated", "disease"].edge_label_index
        )
        return pred


    def prediction(self, data: HeteroData):
        x_dict = {
            "miRNA": self.miRNA_lin(data["miRNA"].x) + self.miRNA_emb(data["miRNA"].node_id),
            "disease": self.disease_emb(data["disease"].node_id)
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier.prediction(
            x_dict["miRNA"],
            x_dict["disease"]
        )
        return pred






