import torch
from torch.nn import ReLU, ModuleDict, Linear, Sequential, LeakyReLU, Dropout, Embedding, Tanh, Parameter
from torch_geometric import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, RGCNConv
import torch.nn.functional as F


class RDGCNModel_v2(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        x_dict = self.encoder(data)
        res = self.decoder(data, x_dict)
        return res


class RDGCNEncoder_v2(torch.nn.Module):
    def __init__(self, data, in_dims, out_dims, slope):
        super().__init__()
        self.updater = FeatureUpdater(data=data, in_dims=in_dims, slope=slope, dropout=0.7)

        # Keep the feature dimensions of input GNN the same
        self.miRNA_emb_lin_update = torch.nn.Linear(1024, in_dims)
        self.miRNA_emb_lin = torch.nn.Linear(data["miRNA"].embedding_feature.shape[-1], in_dims)
        self.miRNA_sim_lin = torch.nn.Linear(data["miRNA"].similarity_feature.shape[-1], in_dims)
        self.miRNA_ass_lin = torch.nn.Linear(data["miRNA"].association_feature.shape[-1], in_dims)
        self.disease_sim_lin = torch.nn.Linear(data["disease"].similarity_feature.shape[-1], in_dims)
        self.disease_ass_lin = torch.nn.Linear(data["disease"].association_feature.shape[-1], in_dims)
        self.original_reshape_lin = torch.nn.Linear(in_dims, out_dims)

        # Controls the weight of each feature before input GNN
        self.miRNA_emb_weight_1 = torch.nn.Parameter(torch.ones(1))
        self.miRNA_sim_weight_1 = torch.nn.Parameter(torch.ones(1))
        self.miRNA_ass_weight_1 = torch.nn.Parameter(torch.ones(1))
        self.disease_sim_weight_1 = torch.nn.Parameter(torch.ones(1))
        self.disease_ass_weight_1 = torch.nn.Parameter(torch.ones(1))

        self.miRNA_emb_weight_2 = torch.nn.Parameter(torch.ones(1))
        self.miRNA_sim_weight_2 = torch.nn.Parameter(torch.ones(1))
        self.miRNA_ass_weight_2 = torch.nn.Parameter(torch.ones(1))
        self.disease_sim_weight_2 = torch.nn.Parameter(torch.ones(1))
        self.disease_ass_weight_2 = torch.nn.Parameter(torch.ones(1))

        # Controls the weight of each feature before input GNN
        num_dict = 2
        self.weights_miRNA = Parameter(torch.ones(num_dict) / num_dict)  # 初始化权重为均等分配
        self.weights_disease = Parameter(torch.ones(num_dict) / num_dict)  # 初始化权重为均等分配

        # Instantiate homogeneous GNN:
        self.gnn_original = GNNSAGEConv(in_dims, out_dims, slope)
        self.gnn_updated = GNNSAGEConv(in_dims, out_dims, slope)

        # Convert GNN model into a heterogeneous variant:
        self.gnn_original = to_hetero(self.gnn_original, metadata=data.metadata())
        self.gnn_updated = to_hetero(self.gnn_updated, metadata=data.metadata())

    def forward(self, data: HeteroData):
        # Feature dimension reshape to same
        miRNA_emb = self.miRNA_emb_lin(data["miRNA"]["embedding_feature"])
        miRNA_sim = self.miRNA_sim_lin(data["miRNA"]["similarity_feature"])
        miRNA_ass = self.miRNA_ass_lin(data["miRNA"]["association_feature"])
        disease_sim = self.disease_sim_lin(data["disease"]["similarity_feature"])
        disease_ass = self.disease_ass_lin(data["disease"]["association_feature"])
        x_dict_original = {
            "miRNA": miRNA_emb * self.miRNA_emb_weight_1
                     + miRNA_sim * self.miRNA_sim_weight_1
                     + miRNA_ass * self.miRNA_ass_weight_1,
            "disease": disease_sim * self.disease_sim_weight_1
                       + disease_ass * self.disease_ass_weight_1
        }

        # Original Feature dic
        x_dict_original_reshaped = {
            "miRNA": self.original_reshape_lin(x_dict_original["miRNA"]),
            "disease": self.original_reshape_lin(x_dict_original["disease"]),
        }
        # # Aggregated Original Feature dic
        # x_dict_original_agg = self.gnn_original(x_dict_original, data.edge_index_dict)

        updated_data = self.updater(data)
        miRNA_emb_update = self.miRNA_emb_lin_update(updated_data["miRNA"]["embedding_feature"])
        miRNA_sim_update = self.miRNA_sim_lin(updated_data["miRNA"]["similarity_feature"])
        miRNA_ass_update = self.miRNA_ass_lin(updated_data["miRNA"]["association_feature"])
        disease_sim_update = self.disease_sim_lin(updated_data["disease"]["similarity_feature"])
        disease_ass_update = self.disease_ass_lin(updated_data["disease"]["association_feature"])
        x_dict_update = {
            "miRNA": miRNA_emb_update * self.miRNA_emb_weight_2
                     + miRNA_sim_update * self.miRNA_sim_weight_2
                     + miRNA_ass_update * self.miRNA_ass_weight_2,
            "disease": disease_sim_update * self.disease_sim_weight_2
                       + disease_ass_update * self.disease_ass_weight_2
        }
        # Aggregated Updated Feature dic
        x_dict_update_agg = self.gnn_updated(x_dict_update, data.edge_index_dict)

        # 3 dicts weights
        dict_list = [x_dict_original_reshaped, x_dict_update_agg]
        weights_miRNA = F.softmax(self.weights_miRNA, dim=0)
        weights_disease = F.softmax(self.weights_disease, dim=0)

        # Aggregate 3 features
        aggregated_dict = {'miRNA': None, 'disease': None}
        aggregated_miRNA = sum([dict_list[i]['miRNA'] * weights_miRNA[i] for i in range(2)])
        aggregated_dict['miRNA'] = aggregated_miRNA
        aggregated_disease = sum([dict_list[i]['disease'] * weights_disease[i] for i in range(2)])
        aggregated_dict['disease'] = aggregated_disease

        return aggregated_dict


class RDGCNDecoder_v2(torch.nn.Module):
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
        self.conv1 = SAGEConv(in_channels=in_channels, out_channels=out_channels )
        self.conv2 = SAGEConv(in_channels=out_channels * 2, out_channels=out_channels)
        self.LeakyReLU = LeakyReLU(slope)
        self.Dropout = Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.LeakyReLU(x)
        x = self.Dropout(x)
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
                            Tanh(),
                            Dropout(dropout)
                        )
                    else:
                        self.feature_updaters[node_type][feature_key] = Sequential(
                            Linear(feature_data.shape[1], feature_data.shape[1]),
                            Tanh(),
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
