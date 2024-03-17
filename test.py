from RDGCN_Dataset import RDGCNDataset

# for node_type in data.node_types:
#     print(f"Node type: {node_type}")
#
#     # 获取所有节点特征
#     for feature_key, feature_value in data[node_type].items():
#         if feature_key != "node_id":
#             print(f"  Feature: {feature_key}")
#             print(f"    Shape: {feature_value.size()}")

# def generate_feature_dim(data, hidden_dim, output_dim):
#     feature_dims = {}
#     for node_type in data.node_types:
#         feature_dims[node_type] = {}
#         for feature_key, feature_value in data[node_type].items():
#             if feature_key != "node_id":
#                 input_dim = feature_value.size(1)
#                 feature_dims[node_type][feature_key] = {
#                     'input_dim': input_dim,
#                     'hidden_dim': hidden_dim,
#                     'output_dim': output_dim
#                 }
#
#     return feature_dims
#
# # 使用示例
# hidden_dim = 128  # 手动设置的隐藏层维度
# output_dim = 64   # 手动设置的输出层维度
#
# # 自动生成feature_dims
# feature_dims = generate_feature_dim(data, hidden_dim, output_dim)
# print(feature_dims)
#
# for node_type, features in feature_dims.items():
#     for feature_name, dims in features.items():
#         print(node_type, " ", feature_name)

dataset = RDGCNDataset(root="./data")
data = dataset[0]
# print(data)

# for node_type in data.node_types:
#     print("Node ", node_type)
#     for feature_key, feature_value in data[node_type].items():
#         if feature_key != "node_id":
#             print(feature_key, feature_value.shape[1])


# import numpy as np
#
# # 替换为你的 .npy 文件路径
# file_path = 'data/miRNA_embedding/hsa-mir-16.npy'
#
# # 加载 .npy 文件
# data = np.load(file_path)
#
# # 打印数据的形状和数据类型
# print(f'Shape of the data: {data.shape}')
# print(f'Data type of the data: {data.dtype}')
#
# # 打印数据的前几个元素（例如前五个）
# print('First few elements of the data:')
# print(data[:5])

# import os
#
# # 获取当前目录
# current_directory = os.getcwd()
#
# # 设置result目录的路径
# result_directory = os.path.join(current_directory, 'result')
#
# # 检查result目录是否存在
# if not os.path.exists(result_directory):
#
#     os.makedirs(result_directory) # 创建result目录
#     os.makedirs(os.path.join(result_directory, 'result_0'))
# else:  # result 目录存在


import os


