import numpy as np
import random
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from RDGCN_Dataset import RDGCNDataset
from models.model_RDGCN_v5 import RDGCNModel_v5, RDGCNEncoder_v5, RDGCNDecoder_v5
from models.model_RDGCN_v6 import RDGCNModel_v6, RDGCNEncoder_v6, RDGCNDecoder_v6
from models.model_RDGCN_v7 import RDGCNModel_v7, RDGCNDecoder_v7, RDGCNEncoder_v7

from utiles import plot_ROC, plot_PR, get_save_dir


def RDGCN(epochs_num,
          n_splits,
          batch_size,
          lr,
          weight_decay,
          in_dims,
          out_dims,
          slope,
          dropout,
          disjoint_train_ratio,
          aggr,
          random_seed):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print("Device: {}".format(device))
    print()

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device == "cuda":
        torch.cuda.manual_seed(random_seed)

    dataset = RDGCNDataset(root="./data")
    data = dataset[0]
    print(data)

    # After transform
    # 1. get edge_label and edge_label_index. edge_index = 5430*0.7 = 3801
    # 2. neg_sampling
    transform = RandomLinkSplit(
        num_val=0,
        num_test=0,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=1.0,
        is_undirected=True,
        add_negative_train_samples=True,
        edge_types=("miRNA", "associated", "disease"),
        rev_edge_types=("disease", "rev_associated", "miRNA"),
    )
    data, _, _ = transform(data)

    test_results_matrix, model_entity = cross_validation_with_val_set(
        data=data,
        device=device,
        epochs_num=epochs_num,
        n_splits=n_splits,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        in_dims=in_dims,
        out_dims=out_dims,
        slope=slope,
        dropout=dropout,
        aggr=aggr,
    )

    print("Average accuracy: ", test_results_matrix[0])
    print("Average precision: ", test_results_matrix[1])
    print("Average recall: ", test_results_matrix[2])
    print("Average F1 score:", test_results_matrix[3])
    print("Average ROC AUC:", test_results_matrix[4])
    print("Average PR AUC:", test_results_matrix[5])

    fold, model, all_ground_truth, all_pred, acc, precision, precision, f1, roc_auc, pr_auc = model_entity
    print(f"The ROC of beat model is {roc_auc}")
    save_dir = get_save_dir(base_dir_name="result")
    torch.save(model.state_dict(), f'{save_dir}/model_state_dict.pth')

    plot_ROC(all_ground_truth, all_pred, save_dir)
    plot_PR(all_ground_truth, all_pred, save_dir)

    # 将参数写入到文本文件
    params_str = f"""
    Parameter setting
        Epochs Number: {epochs_num}
        Number of Splits: {n_splits}
        Batch Size: {batch_size}
        Learning Rate: {lr}
        Weight Decay: {weight_decay}
        Input Dimensions: {in_dims}
        Out_dims: {out_dims}
        Slope: {slope}
        Dropout: {dropout}
        Disjoint_train_ratio: {disjoint_train_ratio}
        Aggr: {aggr}
        Random Seed: {random_seed}
    
    5 fold cross validation result
        Average accuracy: {test_results_matrix[0]}
        Average precision: {test_results_matrix[1]}
        Average recall: {test_results_matrix[2]}
        Average F1 score: {test_results_matrix[3]}
        Average ROC AUC: {test_results_matrix[4]}
        Average PR AUC: {test_results_matrix[5]}
    
    Best model performance
        Accuracy:{acc}
        Precision:{precision}
        Recall:{precision}
        F1 score:{f1}
        ROC AUC:{roc_auc}
        PR AUC:{pr_auc}
    """
    with open(f'{save_dir}/model_summary.txt', 'w') as file:
        file.write(params_str)

def cross_validation_with_val_set(data,
                                  device,
                                  epochs_num,
                                  n_splits,
                                  batch_size,
                                  lr,
                                  weight_decay,
                                  in_dims,
                                  out_dims,
                                  slope,
                                  dropout,
                                  aggr,
                                  ):
    edge_label_index = data["miRNA", "associated", "disease"].edge_label_index
    edge_label_index_transposed = edge_label_index.transpose(0, 1)  # change shape from [2, 3258] to [3258, 2]
    edge_label = data["miRNA", "associated", "disease"].edge_label  # no need to convert the shape

    fold = 0
    test_results_matrix = np.zeros(
        (n_splits, 6))  # store evaluation parameters of the model [acc, precision, recall, f1, roc_auc, pr_auc]
    model_matrix = []  # 2D list. [fold, auc, model]
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
            num_neighbors=[-1] * 3,  # The number of neighbors to sample for each node in each iteration
            edge_label_index=(("miRNA", "associated", "disease"), edge_label_index_train),
            edge_label=edge_label_train,
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[-1] * 3,  # The number of neighbors to sample for each node in each iteration
            edge_label_index=(("miRNA", "associated", "disease"), edge_label_index_test),
            edge_label=edge_label_test,
            batch_size=batch_size,
            shuffle=True,
        )

        # model = Model_SAGEConv(data=data, miRNA_features=miRNA_features, disease_features=disease_features,
        #                        hidden_channels=hidden_channels)

        # model = RDGCNModel(RDGCNEncoder(data=data, in_dims=in_dims, out_dims=64, slope=0.2, dropout=dropout, aggr=aggr),
        #                    RDGCNDecoder())

        # model = RDGCNModel_v2(RDGCNEncoder_v2(data=data, in_dims=in_dims, out_dims=64, slope=0.2),
        #                    RDGCNDecoder_v2())

        # model = RDGCNModel_v3(RDGCNEncoder_v3(data=data, in_dims=in_dims, out_dims=64, slope=0.2),
        #                       RDGCNDecoder_v3())

        # model = RDGCNModel_v4(RDGCNEncoder_v4(data=data, in_dims=in_dims, out_dims=out_dims, slope=slope, dropout=dropout),
        #                       RDGCNDecoder_v4())

        # model = RDGCNModel_v5(
        #     RDGCNEncoder_v5(data=data, in_dims=in_dims, out_dims=out_dims, slope=slope, dropout=dropout, aggr=aggr),
        #     RDGCNDecoder_v5())

        # model = RDGCNModel_v6(
        #     RDGCNEncoder_v6(data=data, in_dims=in_dims, out_dims=out_dims, slope=slope, dropout=dropout, aggr=aggr),
        #     RDGCNDecoder_v6())

        model = RDGCNModel_v7(
            RDGCNEncoder_v7(data=data, in_dims=in_dims, out_dims=out_dims, slope=slope, dropout=dropout, aggr=aggr),
            RDGCNDecoder_v7())

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1, epochs_num + 1):
            train_loss, acc, precision, recall, f1, roc_auc, pr_auc = train(model, optimizer, train_loader, device)
            print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}")
            print(f"Train: Correct predictions: {acc:.4f}, F1 score: {f1:.4f}, ROC: {roc_auc:.4f}, PR: {pr_auc:.4f}")
            if epoch % 10 == 0:
                acc, precision, recall, f1, roc_auc, pr_auc, model, all_ground_truth, all_pred = test(model,
                                                                                                      test_loader,
                                                                                                      device)
                print(f"Test: Correct predictions: {acc:.4f}, F1 score: {f1:.4f}, ROC: {roc_auc:.4f}, PR: {pr_auc:.4f}")
                print()

        acc, precision, recall, f1, roc_auc, pr_auc, model, all_ground_truth, all_pred = test(model, test_loader,
                                                                                              device)
        print(f"Final Test: Correct predictions: {acc:.4f}, F1 score: {f1:.4f}, ROC: {roc_auc:.4f}, PR: {pr_auc:.4f}")
        test_results_matrix[fold - 1, :] = acc, precision, recall, f1, roc_auc, pr_auc
        model_matrix.append([fold - 1, model, all_ground_truth, all_pred, acc, precision, recall, f1, roc_auc,
                             pr_auc])  # add model in model_matrix
        print(f"=============================== End of fold {fold} ===============================")

    average_test_results = np.mean(test_results_matrix, axis=0)
    model_matrix.sort(key=lambda x: x[8], reverse=True)

    return average_test_results, model_matrix[0]


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

    return acc, precision, recall, f1, roc_auc, pr_auc, model, all_ground_truth, all_pred
