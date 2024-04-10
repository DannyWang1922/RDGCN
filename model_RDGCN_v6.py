import torch
from torch.nn import ReLU, ModuleDict, Linear, Sequential, LeakyReLU, Dropout, Embedding
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, RGCNConv
import torch.nn.functional as F

class RDGCNModel_v6(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        x_dict = self.encoder(data)
        res = self.decoder(data, x_dict)
        return res


class RDGCNEncoder_v6(torch.nn.Module):
    def __init__(self, data, in_dims, out_dims, slope, dropout, aggr):
        super().__init__()

        # Keep the feature dimensions of input GNN the same
        self.miRNA_emb_lin = torch.nn.Linear(96000, in_dims)
        self.miRNA_sim_lin = torch.nn.Linear(data["miRNA"].similarity_feature.shape[-1], in_dims)
        self.miRNA_ass_lin = torch.nn.Linear(data["miRNA"].association_feature.shape[-1], in_dims)
        self.disease_sim_lin = torch.nn.Linear(data["disease"].similarity_feature.shape[-1], in_dims)
        self.disease_ass_lin = torch.nn.Linear(data["disease"].association_feature.shape[-1], in_dims)

        # Controls the weight of each feature before input GNN
        self.miRNA_weights = torch.nn.Parameter(torch.ones(3) / 3)
        self.disease_weights = torch.nn.Parameter(torch.ones(2) / 2)

        # Instantiate homogeneous GNN:
        self.gnn = GNNSAGEConv(in_dims, out_dims, slope, aggr)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

    def forward(self, data: HeteroData):
        miRNA_weights = F.softmax(self.miRNA_weights, dim=0)
        disease_weights = F.softmax(self.disease_weights, dim=0)

        miRNA_sim = self.miRNA_sim_lin(data["miRNA"]["similarity_feature"]) * miRNA_weights[0]
        miRNA_ass = self.miRNA_ass_lin(data["miRNA"]["association_feature"]) * miRNA_weights[1]
        miRNA_emb = self.miRNA_emb_lin(data["miRNA"]["embedding_feature"]) * miRNA_weights[2]

        disease_sim = self.disease_sim_lin(data["disease"]["similarity_feature"]) * disease_weights[0]
        disease_ass = self.disease_ass_lin(data["disease"]["association_feature"]) * disease_weights[1]

        x_dict = {
            "miRNA": miRNA_emb + miRNA_sim + miRNA_ass,
            "disease": disease_sim + disease_ass
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        return x_dict

class RDGCNDecoder_v6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = F.sigmoid

    def forward(self, data, x_dict):
        x_miRNA = x_dict["miRNA"]
        x_disease = x_dict["disease"]
        edge_label_index = data["miRNA", "associated", "disease"].edge_label_index

        # Convert node embeddings to edge-level representations:
        edge_feature_RNA = x_miRNA[edge_label_index[0]]
        edge_feature_disease = x_disease[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        res = (edge_feature_RNA * edge_feature_disease).sum(dim=-1)
        return res

    def all_pairs_scores(self, x_miRNA, x_disease):
        scores_matrix = torch.matmul(x_miRNA, x_disease.t())
        scores_matrix = self.activation(scores_matrix)
        return scores_matrix


class GNNSAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, slope, aggr):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=in_channels, out_channels=out_channels * 4, aggr=aggr)
        self.conv2 = SAGEConv(in_channels=out_channels * 4, out_channels=out_channels * 2, aggr=aggr)
        self.conv3 = SAGEConv(in_channels=out_channels * 2, out_channels=out_channels, aggr=aggr)
        self.LeakyReLU = LeakyReLU(slope)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.LeakyReLU(x)
        x = self.conv2(x, edge_index)
        x = self.LeakyReLU(x)
        x = self.conv3(x, edge_index)
        x = self.LeakyReLU(x)
        return x