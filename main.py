import time
import numpy as np
import random

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch import tensor
from torch.optim import Adam

from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader

from torch_geometric.loader import LinkLoader, LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from RDGCN_Dataset import RDGCNDataset
from model_SAGEConv import Model as Model_SAGEConv
from utiles import plot_ROC, plot_PR


def RDGCN(epochs_num, random_seed):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device == "cuda":
        torch.cuda.manual_seed(random_seed)

    dataset = RDGCNDataset(root="./data")
    data = dataset[0]
    # print(data)

    # After transform
    # 1. get edge_label and edge_label_index. edge_index = 5430*0.7 = 3801
    # 2. neg_sampling
    transform = RandomLinkSplit(
        num_val=0,
        num_test=0,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
        is_undirected=True,
        add_negative_train_samples=True,
        edge_types=("miRNA", "associated", "disease"),
        rev_edge_types=("disease", "rev_associated", "miRNA"),
    )
    data, _, _ = transform(data)

    test_results_matrix = cross_validation_with_val_set(data, device, epochs_num)

    print("Average accuracy: ", test_results_matrix[0])
    print("Average precision: ", test_results_matrix[1])
    print("Average recall: ", test_results_matrix[2])
    print("Average F1 score:", test_results_matrix[3])
    print("Average ROC AUC:", test_results_matrix[4])
    print("Average PR AUC:", test_results_matrix[5])

def cross_validation_with_val_set(data,
                                  device,
                                  epochs_num,
                                  n_splits=5,
                                  batch_size=256,
                                  lr=0.001,
                                  weight_decay=0.01,
                                  miRNA_features=96000,
                                  disease_features=383,
                                  hidden_channels=256):

    edge_label_index = data["miRNA", "associated", "disease"].edge_label_index
    edge_label_index_transposed = edge_label_index.transpose(0, 1)  # change shape from [2, 3258] to [3258, 2]
    edge_label = data["miRNA", "associated", "disease"].edge_label  # no need to convert the shape

    fold = 0
    test_results_matrix = np.zeros((5, 6))
    skf = StratifiedKFold(n_splits=n_splits)

    for train_index, test_index in skf.split(edge_label_index_transposed, edge_label):  # 5 fold
        fold += 1
        print("Fold: ", fold)

        edge_label_index_train = edge_label_index_transposed[train_index]
        edge_label_index_test = edge_label_index_transposed[test_index]

        edge_label_index_train = edge_label_index_train.transpose(0, 1)  # change shape back from [n, 2] to [2, n]
        edge_label_index_test = edge_label_index_test.transpose(0, 1)

        edge_label_train, edge_label_test = edge_label[train_index], edge_label[test_index]

        train_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[30] * 2,  # The number of neighbors to sample for each node in each iteration
            edge_label_index=(("miRNA", "associated", "disease"), edge_label_index_train),
            edge_label=edge_label_train,
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[30] * 2,  # The number of neighbors to sample for each node in each iteration
            edge_label_index=(("miRNA", "associated", "disease"), edge_label_index_test),
            edge_label=edge_label_test,
            batch_size=batch_size,
            shuffle=True,
        )

        model = Model_SAGEConv(data=data, miRNA_features=miRNA_features, disease_features=disease_features, hidden_channels=hidden_channels)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1, epochs_num + 1):
            train_loss, acc, precision, recall, f1, roc_auc, pr_auc = train(model, optimizer, train_loader, device)
            print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}")
            print(f"Train: Correct predictions: {acc:.4f}, F1 score: {f1:.4f}, ROC: {roc_auc:.4f}, PR: {pr_auc:.4f}")

        print()
        acc, precision, recall, f1, roc_auc, pr_auc = test(model, test_loader, device)
        print(f"Test: Correct predictions: {acc:.4f}, F1 score: {f1:.4f}, ROC: {roc_auc:.4f}, PR: {pr_auc:.4f}")
        test_results_matrix[fold-1, :] = acc, precision, recall, f1, roc_auc, pr_auc
        print()
        print()

    average_test_results = np.mean(test_results_matrix, axis=0)

    return average_test_results



def train(model, optimizer, train_loader, device):
    model.train()
    total_loss = total_examples = 0
    all_pred = []
    all_ground_truth = []
    for train_data in train_loader:
        optimizer.zero_grad()
        train_data.to(device)
        pred = model(train_data)
        ground_truth = train_data["miRNA", "associated", "disease"].edge_label

        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()  # update parameter

        total_loss += loss.item() * ground_truth.size(0)
        total_examples += ground_truth.size(0)

        all_pred.append(pred)
        all_ground_truth.append(ground_truth)

    all_pred = torch.cat(all_pred, dim=0).cpu().detach().numpy()
    all_ground_truth = torch.cat(all_ground_truth, dim=0).cpu().detach().numpy()
    threshold = 0.5
    all_binary_pred = (all_pred > threshold).astype(int)

    num_correct = (all_binary_pred == all_ground_truth).sum()
    num_samples = len(all_binary_pred)
    acc = num_correct / num_samples
    f1 = f1_score(all_ground_truth, all_binary_pred)  # needs binary pred
    precision = precision_score(all_ground_truth, all_binary_pred)
    recall = recall_score(all_ground_truth, all_binary_pred)

    roc_auc = roc_auc_score(all_ground_truth, all_pred)  # needs logistic pred
    pr_auc = average_precision_score(all_ground_truth, all_pred)

    train_loss = total_loss / total_examples

    return train_loss, acc, precision, recall, f1, roc_auc, pr_auc


def test(model, test_loader, device):
    model.eval()
    all_pred = []
    all_ground_truth = []
    for test_data in test_loader:
        with torch.no_grad():
            test_data.to(device)
            pred = (model(test_data))
            ground_truth = test_data["miRNA", "associated", "disease"].edge_label

            all_pred.append(pred)
            all_ground_truth.append(ground_truth)

    all_pred = torch.cat(all_pred, dim=0).cpu().numpy()
    all_ground_truth = torch.cat(all_ground_truth, dim=0).cpu().numpy()
    threshold = 0.5
    all_binary_pred = (all_pred > threshold).astype(int)

    num_correct = (all_binary_pred == all_ground_truth).sum()
    num_samples = len(all_binary_pred)

    acc = num_correct / num_samples
    precision = precision_score(all_ground_truth, all_binary_pred)
    recall = recall_score(all_ground_truth, all_binary_pred)
    f1 = f1_score(all_ground_truth, all_binary_pred)  # needs binary pred

    roc_auc = roc_auc_score(all_ground_truth, all_pred)  # needs logistic pred
    pr_auc = average_precision_score(all_ground_truth, all_pred)

    return acc, precision, recall, f1, roc_auc, pr_auc


if __name__ == '__main__':
    RDGCN(2, 126)
