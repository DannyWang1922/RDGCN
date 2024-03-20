import numpy as np
import pandas as pd
import torch
from pandas import read_csv
from torch_geometric.transforms import RandomLinkSplit
from RDGCN_Dataset import RDGCNDataset
import warnings

from model_RDGCN import RDGCNModel, RDGCNEncoder, RDGCNDecoder
from utiles import load_txt_to_list, get_save_dir, get_all_edge_index

warnings.filterwarnings("ignore")

dataset = RDGCNDataset(root="./")
data = dataset[0]

# Step1: Construct ground truth array ============================================================
miRNA_idx = data["miRNA", "associated", "disease"].edge_index[0]
disease_idx = data["miRNA", "associated", "disease"].edge_index[1]

num_miRNA = data["miRNA"]["node_id"].shape[0]
num_disease = data["disease"]["node_id"].shape[0]
ground_truth_arr = np.zeros((num_miRNA, num_disease))

for i in range(len(miRNA_idx)):
    miRNA = miRNA_idx[i]
    disease = disease_idx[i]
    ground_truth_arr[miRNA, disease] = 1

# Step2: Construct all edge index ============================================================
# 生成所有miRNA节点和disease节点之间的边索引
miRNA_indices = torch.arange(num_miRNA).unsqueeze(1).repeat(1, num_disease).view(-1)
disease_indices = torch.arange(num_disease).unsqueeze(0).repeat(num_miRNA, 1).view(-1)

# 创建完整的边索引
full_edge_index = torch.stack([miRNA_indices, disease_indices], dim=0)

full_edge_data = data
full_edge_data['miRNA', 'associated', 'disease'].edge_index = full_edge_index  # 将完整的边索引添加到数据对象中

# Step3: init the model and run it ============================================================
model = RDGCNModel(RDGCNEncoder(data=full_edge_data, in_dims=256, out_dims=64, slope=0.2),
                   RDGCNDecoder())
model.load_state_dict(torch.load("result/result_0/model_state_dict_0.9085105741547458.pth"))


def pred(data, model):
    model.eval()
    with torch.no_grad():
        x_dict = model.encoder(data)
        x_miRNA = x_dict["miRNA"]
        x_disease = x_dict["disease"]
        pred_score = model.decoder.all_pairs_scores(x_miRNA, x_disease).numpy()

    threshold = 0.5
    pred_link = (pred_score > threshold).astype(int)

    df = pd.DataFrame(columns=['miRNA_ID', 'Disease_ID'])
    for miRNA_idx in range(pred_score.shape[0]):
        for disease_idx in range(pred_score.shape[1]):
            if pred_link[miRNA_idx][disease_idx] == 1:
                score = pred_score[miRNA_idx][disease_idx]
                association = {'miRNA_ID': miRNA_idx + 1, 'Disease_ID': disease_idx + 1,
                               "Prediction score": score}  # index +1 to match the dataset index
                df = pd.concat([df, pd.DataFrame([association])], ignore_index=True)

    # df = df.replace({"miRNA": miRNA_dic, "Disease": disease_dic})
    df.to_csv(f"{save_dir}/1.Pred_score_idx.csv", index=False)
    return pred_link


save_dir = get_save_dir(base_dir_name="prediction_result")
pred_array = pred(full_edge_data, model)  # (495, 383)

# Step4: Compare with ground truth array using id ============================================================
num_association_pred = np.count_nonzero(pred_array == 1)
print("The number of associations in prediction: ", num_association_pred)

equal_elements_count = np.sum(ground_truth_arr == pred_array)
print("The number of equal elements is: ", equal_elements_count, " The accuracy is :",
      (equal_elements_count / (495 * 383)), "\n")

# Step5: Case study using HMDD v2.0, dbDEMC and miR2Disease database  =======================================================
# Case study HMDD
HMDD_association = load_txt_to_list("data/miRNA_disease/miRNA_disease_association.txt")
df_HMDD = pd.DataFrame(HMDD_association, columns=['miRNA_ID', 'Disease_ID'])
df_HMDD['miRNA_ID'] = df_HMDD['miRNA_ID'].astype(int)
df_HMDD['Disease_ID'] = df_HMDD['Disease_ID'].astype(int)

df_pred = pd.read_csv(f"{save_dir}/1.Pred_score_idx.csv")  # ID start with 1
df_pred = pd.merge(df_pred, df_HMDD, on=['miRNA_ID', 'Disease_ID'], how='left', indicator=True)
df_pred['HMDD'] = df_pred['_merge'] == 'both'

matches_count = df_pred['_merge'].value_counts().get('both', 0)
print(f"The number of same miRNA-disease associations in HMDD and prediction is : {matches_count}")
df_pred.drop(columns='_merge', inplace=True)

# Case study dbDEMC
dbDEMC = load_txt_to_list("data/miRNA_disease/miRNA_disease_association_dbDEMC.txt")
df_dbDEMC = pd.DataFrame(dbDEMC, columns=['miRNA_ID', 'Disease_ID'])
df_dbDEMC['miRNA_ID'] = df_HMDD['miRNA_ID'].astype(int)
df_dbDEMC['Disease_ID'] = df_HMDD['Disease_ID'].astype(int)

df_pred = pd.merge(df_pred, df_dbDEMC, on=['miRNA_ID', 'Disease_ID'], how='left', indicator=True)
df_pred['dbDEMC'] = df_pred['_merge'] == 'both'

matches_count = df_pred['_merge'].value_counts().get('both', 0)
print(f"The number of same miRNA-disease associations in dbDEMC is : {matches_count}")
df_pred.drop(columns='_merge', inplace=True)

# Case study miR2Disease
miR2Disease = load_txt_to_list("data/miRNA_disease/miRNA_disease_association_miR2Disease.txt")
df_miR2Disease = pd.DataFrame(miR2Disease, columns=['miRNA_ID', 'Disease_ID'])
df_miR2Disease['miRNA_ID'] = df_HMDD['miRNA_ID'].astype(int)
df_miR2Disease['Disease_ID'] = df_HMDD['Disease_ID'].astype(int)
df_pred = pd.merge(df_pred, df_miR2Disease, on=['miRNA_ID', 'Disease_ID'], how='left', indicator=True)
df_pred['miR2Disease'] = df_pred['_merge'] == 'both'

matches_count = df_pred['_merge'].value_counts().get('both', 0)
print(f"The number of same miRNA-disease associations in miR2Disease is : {matches_count}")
df_pred.drop(columns='_merge', inplace=True)

# save the case study result
df_pred.to_csv(f"{save_dir}/2.Case_study_idx.csv", index=False)

# Step6: Case study result process  ============================================================================
# Calculate the number of each Disease using groupby and create a new column using transform
df_pred = read_csv(f"{save_dir}/2.Case_study_idx.csv")
df_pred['Disease_Count'] = df_pred.groupby('Disease_ID')['Disease_ID'].transform('count')

# Sort the DataFrame in descending order by Disease_Count and score
df_sorted_by_disease_num = df_pred.sort_values(by=['Disease_Count', 'Prediction score'], ascending=[False, False])
df_sorted_by_disease_num = df_sorted_by_disease_num.reset_index(drop=True)
df_sorted_by_disease_num.to_csv(f"{save_dir}/3.Case_study_idx_sorted.csv", index=False)

# ID to name dict mapping
miRNA_idx = load_txt_to_list("data/miRNA_disease/miRNA_idx.txt")
df_miRNA_idx = pd.DataFrame(miRNA_idx, columns=["miRNA_ID", "miRNA"])
df_miRNA_idx_mapping = df_miRNA_idx.set_index("miRNA_ID")
index_series = df_miRNA_idx_mapping['miRNA']
miRNA_dict = index_series.to_dict()

disease_idx = load_txt_to_list("data/miRNA_disease/disease_idx.txt")
df_disease_idx = pd.DataFrame(disease_idx, columns=["Disease_ID", "Disease"])
df_disease_idx_mapping = df_disease_idx.set_index("Disease_ID")
index_series = df_disease_idx_mapping['Disease']
disease_dict = index_series.to_dict()

# pred res index to name
df_pred_idx_sorted = read_csv(f"{save_dir}/3.Case_study_idx_sorted.csv")

# convert str to int (the id in df_pre is int type)
miRNA_dict = {int(k): v for k, v in miRNA_dict.items()}
disease_dict = {int(k): v for k, v in disease_dict.items()}

df_pred_sorted = df_pred_idx_sorted.replace({"miRNA_ID": miRNA_dict, "Disease_ID": disease_dict})
df_pred_sorted.rename(columns={"miRNA_ID": "miRNA", "Disease_ID": "Disease"}, inplace=True)
df_pred_sorted.to_csv(f"{save_dir}/4.Case_study_name.csv", index=False)

new_association = df_pred_sorted.loc[(df_pred_sorted["dbDEMC"] == True) | (df_pred_sorted["miR2Disease"] == True)]
new_association.to_csv(f"{save_dir}/5.Case_study_new_association.csv", index=False)
