import matplotlib.pyplot as plt

# Dropout率的列表
dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 每个dropout率对应的平均ROC AUC值的列表
roc_auc = [
    0.9218800131592781,
    0.9251026796767159,
    0.9260523183053019,
    0.9270446906721747,
    0.9283551919796235,
    0.9281571244942327,
    0.9287547185169357,
    0.9279882244810054,
    0.9252756495697797
]

# 创建绘图
fig, ax = plt.subplots(figsize=(10, 7))

# 绘制柱状图
# bars = ax.bar(disjoint_ratios, roc_auc, width=0.05, alpha=1, color="#F3D266", label='ROC AUC', zorder=2.5)
bars = ax.bar(dropout_rate, roc_auc, width=0.05, alpha=1, color=plt.get_cmap('Set1')(4), label='ROC AUC', zorder=2.5)
# 在每个柱子上方显示数值，增大字体大小
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), ha='center', va='bottom', fontsize=15)

ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=1.0, zorder=2)

# 设置字体大小
plt.xticks(dropout_rate, fontsize=15, fontweight='bold')
plt.yticks(fontsize=15, fontweight='bold')

# 设置坐标轴刻度和标签
plt.xlabel("Dropout Rate", fontsize=20)
plt.ylabel("AUC-ROC", fontsize=20)

# # 设置图例
# plt.legend(loc='best', fontsize=15)

# 设置坐标轴范围
plt.ylim(0.90, 0.935)

# # 设置标题
plt.title('Different Dropout rate AUC-ROC Performance', fontsize=20, fontweight='bold')

# 显示图形
plt.show()
plt.show()