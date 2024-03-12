from typing import Callable, List, Optional
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
import pandas as pd
from utiles import load_txt_to_list, matrix_to_pair, get_miRNA_embedding_feature, get_similarity_feature, \
    bulit_integrated_similarity, get_association_feature
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


class RDGCNDataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.root = root
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['miRNA_disease.pt']

    def download(self):
        pass

    def process(self):
        # bulit integrated miRNA similarity and disease similarity
        bulit_integrated_similarity()

        # miRNA node ====================================================================================================
        miRNA = load_txt_to_list('data/miRNA_disease/miRNA_idx.txt')
        miRNA_idx = [int(row[0]) for row in miRNA]  # 495
        miRNA_idx_tensor = torch.LongTensor(miRNA_idx)
        miRNA_idx_tensor = miRNA_idx_tensor - 1

        # disease node ====================================================================================================
        disease = load_txt_to_list('data/miRNA_disease/disease_idx.txt')
        disease_idx = [int(row[0]) for row in disease]  # 383
        disease_idx_tensor = torch.LongTensor(disease_idx)
        disease_idx_tensor = disease_idx_tensor - 1

        # miRNA similarity ====================================================================================================
        miRNA_similarity_matrix = np.loadtxt('data/preprocessed/miRNA_similarity_integrated.txt', delimiter=' ', dtype=float)  # (495, 495)\
        miRNA_similarity_pair_list = matrix_to_pair(miRNA_similarity_matrix)  # 32385  matrix to pair
        miRNA_similarity_edge_index = torch.transpose(torch.tensor(miRNA_similarity_pair_list), 0,
                                                      1)  # list to tensor, transpose tensor
        miRNA_similarity_edge_index = miRNA_similarity_edge_index - 1

        # disease similarity ===================================================================================================
        disease_similarity_matrix = np.loadtxt('data/preprocessed/disease_similarity_integrated.txt', delimiter=' ', dtype=float)  # (383, 383)
        disease_similarity_pair_list = matrix_to_pair(disease_similarity_matrix)
        disease_similarity_edge_index = torch.transpose(torch.tensor(disease_similarity_pair_list), 0,
                                                        1)  # list to tensor, transpose tensor
        disease_similarity_edge_index = disease_similarity_edge_index - 1

        # miRNA disease association =======================================================================================
        miRNA_disease_association = np.loadtxt('data/miRNA_disease/miRNA_disease_association.txt', delimiter='\t',
                                               dtype=int)  # 5430
        miRNA_disease_association_edge_index = torch.transpose(torch.tensor(miRNA_disease_association), 0,
                                                               1)  # list to tensor, transpose tensor
        miRNA_disease_association_edge_index = miRNA_disease_association_edge_index - 1

        # miRNA_feature and disease_feature ============================================================================
        miRNA_embedding_feature = get_miRNA_embedding_feature("data/miRNA_disease/miRNA_idx.txt")

        miRNA_similarity_feature = get_similarity_feature("data/preprocessed/miRNA_similarity_integrated.txt")
        disease_similarity_feature = get_similarity_feature("data/preprocessed/disease_similarity_integrated.txt")

        miRNA_association_feature, disease_association_feature = get_association_feature("data/miRNA_disease/miRNA_disease_association.txt")

        # Construct HeteroData object ================================================================================
        data = HeteroData()
        data["miRNA"].node_id = miRNA_idx_tensor
        data["disease"].node_id = disease_idx_tensor

        data["miRNA"].embedding_feature = miRNA_embedding_feature
        data["miRNA"].similarity_feature = miRNA_similarity_feature
        data["miRNA"].association_feature = miRNA_association_feature

        data["disease"].similarity_feature = disease_similarity_feature
        data["disease"].association_feature = disease_association_feature

        data["miRNA", "similar", "miRNA"].edge_index = miRNA_similarity_edge_index
        data["disease", "similar", "disease"].edge_index = disease_similarity_edge_index
        data["miRNA", "associated", "disease"].edge_index = miRNA_disease_association_edge_index

        # reverse edges from disease to RNA
        data = T.ToUndirected()(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save((self.collate([data])), self.processed_paths[0])

# if __name__ == '__main__':

