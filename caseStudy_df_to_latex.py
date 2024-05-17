import pandas as pd
from pandas import read_csv

df = read_csv("prediction_result/prediction_result_1/3.Case_study_sorted.csv")
df = df.drop(["Prediction score", "Evidence", "Evidence_Count"], axis=1)


def create_evidence(row):
    evidence_list = []
    if row['HMDD']:
        evidence_list.append('HMDD')
    if row['dbDEMC']:
        evidence_list.append('dbDEMC')
    if row['miR2Disease']:
        evidence_list.append('miR2Disease')
    if not evidence_list:
        return 'unconfirmed'

    return '; '.join(evidence_list)


# 应用这个函数到每一行，并创建新的"Evidence"列
df['Evidence'] = df.apply(create_evidence, axis=1)
df = df.drop(["HMDD", "dbDEMC", "miR2Disease"], axis=1)

all_diseases = df['Disease'].unique()[:350]

three_diseases = [all_diseases[i] for i in [0, 5, 7]]
print(three_diseases)
print()
print()

# 过滤出只包含前三种疾病的行
df_top_three = df[df['Disease'].isin(three_diseases)]

# 每个疾病的top50
df_top50 = df_top_three.groupby('Disease').head(50)
df_top50 = df_top50[["Disease", "miRNA", "Evidence"]]

dfs = {disease: data.reset_index(drop=True) for disease, data in df_top50.groupby('Disease')}
df_disease_1 = dfs[three_diseases[0]]
df_disease_2 = dfs[three_diseases[1]]
df_disease_3 = dfs[three_diseases[2]]


df_disease_1 = df_disease_1.drop(["Disease"], axis=1)
df_disease_1_part1 = df_disease_1[:25]
df_disease_1_part2 = df_disease_1.iloc[25:].reset_index(drop=True)
df_disease_1 = pd.concat([df_disease_1_part1, df_disease_1_part2], axis=1)
df_disease_1.columns = ['miRNA', 'Evidence', 'miRNA', 'Evidence']
latex_table_code1 = df_disease_1.to_latex(index=False, header=True, longtable=True)
print(latex_table_code1)
print()
print()
df_disease_2 = df_disease_2.drop(["Disease"], axis=1)
df_disease_2_part1 = df_disease_2[:25]
df_disease_2_part2 = df_disease_2.iloc[25:].reset_index(drop=True)
df_disease_2 = pd.concat([df_disease_2_part1, df_disease_2_part2], axis=1)
df_disease_2.columns = ['miRNA', 'Evidence', 'miRNA', 'Evidence']
latex_table_code2 = df_disease_2.to_latex(index=False, header=True, longtable=True)
print(latex_table_code2)
print()
print()
df_disease_3 = df_disease_3.drop(["Disease"], axis=1)
df_disease_3_part1 = df_disease_3[:25]
df_disease_3_part2 = df_disease_3.iloc[25:].reset_index(drop=True)
df_disease_3 = pd.concat([df_disease_3_part1, df_disease_3_part2], axis=1)
df_disease_3.columns = ['miRNA', 'Evidence', 'miRNA', 'Evidence']
latex_table_code3 = df_disease_3.to_latex(index=False, header=True, longtable=True)
print(latex_table_code3)






