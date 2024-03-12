import numpy as np
import torch


def get_association_feature(association_file):
    miRNA_disease_association = np.loadtxt(association_file, delimiter='\t',dtype=int)

    relationship_matrix = np.zeros((495, 383))
    for mirna, disease in miRNA_disease_association:
        # 减1是因为Python的索引从0开始
        relationship_matrix[mirna - 1, disease - 1] = 1

    transposed_matrix = relationship_matrix.T

    miRNA_association_feature = relationship_matrix
    disease_association_feature = transposed_matrix

    return miRNA_association_feature, disease_association_feature





get_association_feature("data/miRNA_disease/miRNA_disease_association.txt")