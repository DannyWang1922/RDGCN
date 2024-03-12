import torch
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

# 假设我们有一个数据集 X 和对应的标签 y
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])

# 创建 KFold 对象实例，设置折数为 k
k = 2
kf = KFold(n_splits=k)

# 使用 KFold 提供的 split 方法来进行数据集的分割
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 在这里，你可以放置你的模型训练和评估代码
    # 例如，使用 X_train, y_train 来训练模型
    # 然后使用 X_test, y_test 来评估模型性能

