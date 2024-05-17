import torch
from torch.nn import ReLU, ModuleDict, Linear, Sequential, LeakyReLU, Dropout, Embedding
from torch_geometric import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, RGCNConv
import torch.nn.functional as F


class RDGCNModel_v3(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        x_dict = self.encoder(data)
        res = self.decoder(data, x_dict)
        return res


class RDGCNEncoder_v3(torch.nn.Module):
    def __init__(self, data, in_dims, out_dims, slope):
        super().__init__()
        self.updater = FeatureUpdater(data=data, in_dims=in_dims, slope=slope, dropout=0.7)

        # Keep the feature dimensions of input GNN the same
        self.miRNA_emb_lin = torch.nn.Linear(96000, out_dims)
        self.miRNA_sim_lin = torch.nn.Linear(data["miRNA"].similarity_feature.shape[-1], out_dims)
        self.miRNA_ass_lin = torch.nn.Linear(data["miRNA"].association_feature.shape[-1], out_dims)
        self.disease_sim_lin = torch.nn.Linear(data["disease"].similarity_feature.shape[-1], out_dims)
        self.disease_ass_lin = torch.nn.Linear(data["disease"].association_feature.shape[-1], out_dims)

        self.miRNA_emb_lin_update = torch.nn.Linear(1024, in_dims)
        self.miRNA_sim_lin_update = torch.nn.Linear(data["miRNA"].similarity_feature.shape[-1], in_dims)
        self.miRNA_ass_lin_update = torch.nn.Linear(data["miRNA"].association_feature.shape[-1], in_dims)
        self.disease_sim_lin_update = torch.nn.Linear(data["disease"].similarity_feature.shape[-1], in_dims)
        self.disease_ass_lin_update = torch.nn.Linear(data["disease"].association_feature.shape[-1], in_dims)

        # Controls the weight of each feature before input GNN
        self.miRNA_weights = torch.nn.Parameter(torch.ones(3)/3)
        self.disease_weights = torch.nn.Parameter(torch.ones(2)/2)

        self.miRNA_weights_update = torch.nn.Parameter(torch.ones(3)/3)
        self.disease_weights_update = torch.nn.Parameter(torch.ones(2)/2)

        self.agg_weights_miRNA = torch.nn.Parameter(torch.ones(2)/2)
        self.agg_weights_disease = torch.nn.Parameter(torch.ones(2)/2)

        # Instantiate homogeneous GNN:
        self.gnn = GNNSAGEConv(in_dims, out_dims, slope)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.gnn_update = GNNSAGEConv(in_dims, out_dims, slope)
        self.gnn_update = to_hetero(self.gnn_update, metadata=data.metadata())

    def forward(self, data: HeteroData):
        miRNA_weights = F.softmax(self.miRNA_weights, dim=0)
        disease_weights = F.softmax(self.disease_weights, dim=0)
        miRNA_weights_update = F.softmax(self.miRNA_weights_update, dim=0)
        disease_weights_update = F.softmax(self.disease_weights_update, dim=0)

        agg_weights_miRNA = F.softmax(self.agg_weights_miRNA, dim=0)
        agg_weights_disease = F.softmax(self.agg_weights_disease, dim=0)

        miRNA_emb = self.miRNA_emb_lin(data["miRNA"]["embedding_feature"]) * miRNA_weights[0]
        miRNA_sim = self.miRNA_sim_lin(data["miRNA"]["similarity_feature"]) * miRNA_weights[1]
        miRNA_ass = self.miRNA_ass_lin(data["miRNA"]["association_feature"]) * miRNA_weights[2]
        disease_sim = self.disease_sim_lin(data["disease"]["similarity_feature"]) * disease_weights[0]
        disease_ass = self.disease_ass_lin(data["disease"]["association_feature"]) * disease_weights[1]
        x_dict = {
            "miRNA": miRNA_emb + miRNA_sim + miRNA_ass,
            "disease": disease_sim + disease_ass
        }

        updated_data = self.updater(data)
        miRNA_emb_update = self.miRNA_emb_lin_update(updated_data["miRNA"]["embedding_feature"]) * miRNA_weights_update[0]
        miRNA_sim_update = self.miRNA_sim_lin_update(updated_data["miRNA"]["similarity_feature"]) * miRNA_weights_update[1]
        miRNA_ass_update = self.miRNA_ass_lin_update(updated_data["miRNA"]["association_feature"]) * miRNA_weights_update[2]
        disease_sim_update = self.disease_sim_lin_update(updated_data["disease"]["similarity_feature"]) * disease_weights_update[0]
        disease_ass_update = self.disease_ass_lin_update(updated_data["disease"]["association_feature"]) * disease_weights_update[1]
        x_dict_update = {
            "miRNA": miRNA_emb_update + miRNA_sim_update + miRNA_ass_update,
            "disease": disease_sim_update + disease_ass_update
        }
        x_dict_update = self.gnn_update(x_dict_update, data.edge_index_dict)


        x_dict_aggregated = {
            "miRNA": x_dict["miRNA"] * agg_weights_miRNA[0] + x_dict_update["miRNA"] * agg_weights_miRNA[1],
            "disease": x_dict["disease"] * agg_weights_disease[0] + x_dict_update["disease"] * agg_weights_disease[1]
        }
        return x_dict_aggregated


class RDGCNDecoder_v3(torch.nn.Module):
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
    def __init__(self, in_channels, out_channels, slope):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=in_channels, out_channels=out_channels * 4)
        self.conv2 = SAGEConv(in_channels=out_channels * 4, out_channels=out_channels * 2)
        self.conv3 = SAGEConv(in_channels=out_channels * 2, out_channels=out_channels)
        self.LeakyReLU = LeakyReLU(slope)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.LeakyReLU(x)
        x = self.conv2(x, edge_index)
        x = self.LeakyReLU(x)
        x = self.conv3(x, edge_index)
        return x


class FeatureUpdater(torch.nn.Module):
    def __init__(self, data, in_dims, slope, dropout):
        super(FeatureUpdater, self).__init__()
        self.feature_updaters = ModuleDict()

        for node_type in data.node_types:
            self.feature_updaters[node_type] = ModuleDict()
            for feature_key, feature_data in data[node_type].items():
                if feature_key != "node_id":
                    if feature_key == "embedding_feature":
                        self.feature_updaters[node_type][feature_key] = Sequential(
                            Linear(feature_data.shape[1], 1024),
                            LeakyReLU(slope),
                            Dropout(dropout)
                        )
                    else:
                        self.feature_updaters[node_type][feature_key] = Sequential(
                            Linear(feature_data.shape[1], feature_data.shape[1]),
                            LeakyReLU(slope),
                            Dropout(dropout)
                        )

    def forward(self, data):
        updated_features = {}
        for node_type in data.node_types:
            updated_features[node_type] = {}
            for feature_type, feature_data in data[node_type].items():
                if feature_type != "node_id" and feature_type != "n_id":
                    # Apply the corresponding feature updater to the feature_data
                    updated_features[node_type][feature_type] = self.feature_updaters[node_type][feature_type](
                        feature_data)
        return updated_features
