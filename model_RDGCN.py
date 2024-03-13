import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, RGCNConv
import torch.nn.functional as F


class RDGCNModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        x_dict = self.encoder(data)
        res = self.decoder(data, x_dict)
        return res


class RDGCNEncoder(torch.nn.Module):
    def __init__(self, data, miRNA_features, disease_features, hidden_channels, out_channels):
        super().__init__()
        self.miRNA_lin = torch.nn.Linear(miRNA_features, hidden_channels)
        self.miRNA_emb = torch.nn.Embedding(data["miRNA"].num_nodes, hidden_channels)

        self.disease_lin = torch.nn.Linear(disease_features, hidden_channels)
        self.disease_emb = torch.nn.Embedding(data["disease"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNNSAGEConv(hidden_channels, out_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

    def forward(self, data: HeteroData):
        x_dict = {
            "miRNA": self.miRNA_lin(data["miRNA"].embedding_feature) + self.miRNA_emb(data["miRNA"].node_id),
            "disease": self.disease_lin(data["disease"].similarity_feature) + self.disease_emb(data["disease"].node_id)
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        return x_dict


class RDGCNDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, x_dict):
        x_miRNA = x_dict["miRNA"]
        x_disease = x_dict["disease"]
        edge_label_index = data["miRNA", "associated", "disease"].edge_label_index

        # Convert node embeddings to edge-level representations:
        edge_feature_RNA = x_miRNA[edge_label_index[0]]
        edge_feature_disease = x_disease[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feature_RNA * edge_feature_disease).sum(dim=-1)

    def prediction(self, x_miRNA, x_disease):
        res = torch.matmul(x_miRNA, x_disease.t())
        return res


class GNNSAGEConv(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, out_channels * 2)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
