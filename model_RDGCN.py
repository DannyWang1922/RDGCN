import torch
from torch.nn import ReLU, ModuleDict, Linear, Sequential, LeakyReLU, Dropout, Embedding
from torch_geometric import nn
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


class GraphSageLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSageLayer, self).__init__()
        self.conv = SAGEConv(in_channels, out_channels, normalize=True)  # 使用内建的SAGEConv层

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x


class RDGCNEncoder(torch.nn.Module):
    def __init__(self, data, in_dims, out_dims, slope):
        super().__init__()
        self.miRNA_emb_lin = torch.nn.Linear(data["miRNA"].embedding_feature.shape[-1], in_dims)
        self.miRNA_sim_lin = torch.nn.Linear(data["miRNA"].similarity_feature.shape[-1], in_dims)
        self.miRNA_ass_lin = torch.nn.Linear(data["miRNA"].association_feature.shape[-1], in_dims)

        self.disease_sim_lin = torch.nn.Linear(data["disease"].similarity_feature.shape[-1], in_dims)
        self.disease_ass_lin = torch.nn.Linear(data["disease"].association_feature.shape[-1], in_dims)

        self.updater = FeatureUpdater(data, in_dims, slope)

        # Instantiate homogeneous GNN:
        self.gnn = GNNSAGEConv(in_dims, out_dims)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

    def forward(self, data: HeteroData):
        updated_data = self.updater(data)

        x_dict = {
            "miRNA": updated_data["miRNA"]["embedding_feature"] + updated_data["miRNA"]["similarity_feature"] + updated_data["miRNA"]["association_feature"],
            "disease": updated_data["disease"]["similarity_feature"] + updated_data["disease"]["association_feature"]
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
        res = (edge_feature_RNA * edge_feature_disease).sum(dim=-1)
        return res

    def prediction(self, x_miRNA, x_disease):
        res = torch.matmul(x_miRNA, x_disease.t())
        return res


class GNNSAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, out_channels * 2)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# class HeteroNodeUpdater(torch.nn.Module):
#     def __init__(self, data: HeteroData, in_channels, out_channels):
#         super(HeteroNodeUpdater, self).__init__()
#         # node_feature_dims: a dictionary mapping node types to their feature dimensions
#         # hidden_dim: the dimension of the hidden layer for all FNNs
#         # out_dims: a dictionary mapping node types to their output feature dimensions
#
#         self.updaters = ModuleDict()
#         for node_type, in_features in data.items():
#             out_features = out_channels.get(node_type, in_features)  # Default to in_features if out_dim is not provided
#             # Define a two-layer FNN for each node type
#             self.updaters[node_type] = Sequential(
#                 Linear(in_features, in_channels),
#                 ReLU(),
#                 Linear(in_channels, out_features)
#             )
#
#     def forward(self, x_dict):
#         # x_dict: a dictionary mapping node types to their feature tensors
#         updated_x_dict = {}
#         for node_type, x in x_dict.items():
#             updater = self.updaters[node_type]
#             updated_x_dict[node_type] = updater(x)
#         return updated_x_dict

class FeatureUpdater(torch.nn.Module):
    def __init__(self, data, in_dims, slope):
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
                            Linear(1024, in_dims),
                            Dropout(0.7)
                        )
                    else:
                        self.feature_updaters[node_type][feature_key] = Sequential(
                            Linear(feature_data.shape[1], in_dims),
                            LeakyReLU(slope),
                            Dropout(0.7)
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


class NodeUpdate(torch.nn.Module):
    def __init__(self, feature_size, slope):
        super(NodeUpdate, self).__init__()

        self.feature_size = feature_size
        self.leakyrelu = LeakyReLU(slope)
        self.linear = torch.nn.Linear(feature_size, feature_size)
        self.dropout = Dropout(0.7)

    def forward(self, nodes):
        h1 = nodes.data['h1']
        h1_new = self.dropout(self.leakyrelu(self.linear(h1)))
        return {'h1': h1_new}


def generate_feature_dims(data, hidden_dim, output_dim):
    feature_dims = {}
    for node_type in data.node_types:
        feature_dims[node_type] = {}
        for feature_key, feature_value in data[node_type].items():
            if feature_key != "node_id":
                input_dim = feature_value.size(1)
                feature_dims[node_type][feature_key] = {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'output_dim': output_dim
                }

    return feature_dims
