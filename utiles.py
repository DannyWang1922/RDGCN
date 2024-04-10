import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, roc_curve, \
    precision_recall_curve


def get_miRNA_embedding_feature(miRNA_idx_file):
    miRNA_idx_file = load_txt_to_list(miRNA_idx_file)
    df_miRNA_idx = pd.DataFrame(miRNA_idx_file, columns=["miRNA_ID", "miRNA"])
    df_miRNA_idx_mapping = df_miRNA_idx.set_index("miRNA_ID")
    index_series = df_miRNA_idx_mapping['miRNA']
    miRNA_dict = index_series.to_dict()

    node_features = []
    for miRNA_idx, miRNA_name in miRNA_dict.items():
        embedding = np.load(f"data/miRNA_embedding/{miRNA_name}.npy", allow_pickle=True)
        if embedding.shape[0] == 3:  # miRNA without sequence
            node_features.append(np.zeros((64, 640)))  # padding with 64 x 640 zero
            print(miRNA_name)
        else:  # miRNA with sequence
            node_features.append(embedding)

    max_length = max([features.shape[0] for features in node_features])
    padded_node_features = []
    for features in node_features:
        padding_length = max_length - features.shape[0]
        padded_features = np.pad(features, ((0, padding_length), (0, 0)))
        padded_node_features.append(padded_features)

    padded_node_features = np.array(padded_node_features)
    node_features_tensor = torch.tensor(padded_node_features, dtype=torch.float32)
    node_features_tensor = node_features_tensor.view(node_features_tensor.size(0), -1)
    return node_features_tensor


def get_similarity_feature(similarity_matrix):
    similarity_matrix = np.loadtxt(similarity_matrix)
    similarity_matrix_float = [[int(item) for item in sublist] for sublist in similarity_matrix]
    similarity_array = np.array(similarity_matrix_float)
    similarity_tensor = torch.tensor(similarity_array, dtype=torch.float32)
    return similarity_tensor


def get_association_feature(association_file):
    miRNA_disease_association = np.loadtxt(association_file, delimiter='\t', dtype=int)
    relationship_matrix = np.zeros((495, 383))
    for mirna, disease in miRNA_disease_association:
        relationship_matrix[mirna - 1, disease - 1] = 1  # Minus 1 because Python's index starts at 0

    transposed_matrix = relationship_matrix.T
    miRNA_association_feature = relationship_matrix
    disease_association_feature = transposed_matrix

    miRNA_association_feature = torch.tensor(miRNA_association_feature, dtype=torch.float32)
    disease_association_feature = torch.tensor(disease_association_feature, dtype=torch.float32)

    return miRNA_association_feature, disease_association_feature


def run_model(test_loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preds = []
    ground_truths = []
    for test_data in test_loader:
        with torch.no_grad():
            test_data = test_data.to(device)
            preds.append(model(test_data))
            ground_truths.append(test_data["miRNA", "associated", "disease"].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

    threshold = 0.5
    binary_pred = (pred > threshold).astype(int)

    return pred, binary_pred, ground_truth


def evaluation_criteria(pred, binary_pred, ground_truth):
    correct = (binary_pred == ground_truth).sum()
    acc = correct / len(binary_pred)
    precision = precision_score(ground_truth, binary_pred)
    recall = recall_score(ground_truth, binary_pred)
    f1 = f1_score(ground_truth, binary_pred)
    auc_roc = roc_auc_score(ground_truth, pred)
    auc_prc = average_precision_score(ground_truth, pred)

    return correct, acc, precision, recall, f1, auc_roc, auc_prc


def plot_ROC(ground_truth, pred, dir):
    fpr, tpr, _ = roc_curve(ground_truth, pred)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='RDGCN')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{dir}/roc_curve.png')
    plt.show()


def plot_PR(ground_truth, pred, dir):
    precision, recall, _ = precision_recall_curve(ground_truth, pred)

    plt.step(recall, precision, color='b', alpha=0.2, where='post', label='RDGCN')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.savefig(f'{dir}/pr_curve.png')
    plt.show()


def load_txt_to_list(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            data.append(line)
    return data


def similarity_matrix_to_similarity_pair(similarity_matrix):
    similarity_pair_list = []
    for i in range(len(similarity_matrix)):
        for j in range(i, len(similarity_matrix)):  # Only work with the upper triangle
            if similarity_matrix[i][j] > 0.5 and similarity_matrix[i][j] !=1:
                similarity_pair_list.append((i + 1, j + 1))
    return similarity_pair_list


def bulit_integrated_similarity():
    # Define functions for Gaussian similarity
    def get_gauss_similarity(matrix, gamma=None):
        if gamma is None:
            norm_squared = np.sum(np.linalg.norm(matrix, axis=1) ** 2)
            gamma = 1.0 / (norm_squared / len(matrix))
        diff_norm_squared = np.sum((matrix[:, np.newaxis] - matrix[np.newaxis, :]) ** 2, axis=2)
        similarity = np.exp(-gamma * diff_norm_squared)
        return similarity

    def integrate_similarity(basic_similarity, weight_matrix, gauss_similarity):
        return np.where(weight_matrix == 1, basic_similarity, gauss_similarity)

    # Load data
    disease_SS1 = np.loadtxt("data/miRNA_disease/disease_SS1.txt")
    disease_SS2 = np.loadtxt("data/miRNA_disease/disease_SS2.txt")
    disease_similarity_weight = np.loadtxt("data/miRNA_disease/disease_similarity_weight.txt")

    miRNA_similarity = np.loadtxt("data/miRNA_disease/miRNA_similarity.txt")
    miRNA_similarity_weight = np.loadtxt("data/miRNA_disease/miRNA_similarity_weight.txt")

    miRNA_disease_association = np.loadtxt("data/miRNA_disease/miRNA_disease_association.txt", dtype=int) - 1

    # Constants
    num_miRNA, num_disease = 495, 383

    # Create association matrix
    miRNA_disease_association_matrix = np.zeros((num_miRNA, num_disease), dtype=float)
    miRNA_disease_association_matrix[miRNA_disease_association[:, 0], miRNA_disease_association[:, 1]] = 1

    # Calculate Gaussian similarities
    gauss_miRNA = get_gauss_similarity(miRNA_disease_association_matrix)
    gauss_disease = get_gauss_similarity(miRNA_disease_association_matrix.T)

    # Integrate similarities with weights
    miRNA_similarity_integrated = integrate_similarity(miRNA_similarity, miRNA_similarity_weight, gauss_miRNA)
    disease_similarity_integrated = integrate_similarity((disease_SS1 + disease_SS2) / 2, disease_similarity_weight,
                                                         gauss_disease)

    np.savetxt('data/preprocessed/miRNA_similarity_integrated.txt', miRNA_similarity_integrated, fmt='%f',
               delimiter=' ')
    np.savetxt('data/preprocessed/disease_similarity_integrated.txt', disease_similarity_integrated, fmt='%f',
               delimiter=' ')


def get_save_dir(base_dir_name):
    # 1. 检查是否存在result目录，若不存在则创建
    if not os.path.exists(base_dir_name):
        os.makedirs(base_dir_name)
        # print(f"The directory '{base_dir_name}' has been created.")

    # 2. 在result目录中检查是否存在任何子目录，若不存在则创建result_0
    subdirectories = [d for d in os.listdir(base_dir_name) if os.path.isdir(os.path.join(base_dir_name, d))]
    if not subdirectories:
        initial_subdir = os.path.join(base_dir_name, f"{base_dir_name}_1")
        os.makedirs(initial_subdir)
        # print(f"The subdirectory '{initial_subdir}' has been created.")
        return initial_subdir  # 直接返回，因为这是第一个创建的子目录

    # 3. 获得索引值最大的目录
    max_index = -1
    for subdir in subdirectories:
        if subdir.startswith(f"{base_dir_name}_"):
            try:
                index = int(subdir.split("_")[-1])
                if index > max_index:
                    max_index = index
            except ValueError:
                # 如果转换失败，忽略这个子目录
                continue

    # 4. 创建result/result_{max_index+1}的目录
    next_index = max_index + 1
    next_subdir = os.path.join(base_dir_name, f"{base_dir_name}_{next_index}")
    os.makedirs(next_subdir)
    # print(f"The subdirectory '{next_subdir}' has been created.")

    # 5. 返回上述目录地址
    return next_subdir

def get_all_edge_index(miRNA_num, disease_num):
    # 生成两个范围的tensor: 0-494 和 0-382
    range1 = torch.arange(0, miRNA_num, dtype=torch.int64)
    range2 = torch.arange(0, disease_num, dtype=torch.int64)

    # 使用meshgrid来获取所有组合
    grid1, grid2 = torch.meshgrid(range1, range2, indexing='ij')

    # 转换为2xn维度的tensor，其中n是所有可能的组合数
    combination_tensor = torch.stack((grid1.flatten(), grid2.flatten()), dim=0)

    return combination_tensor

