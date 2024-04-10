import matplotlib.pyplot as plt

disjoint_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
roc_auc = [0.9036, 0.9186, 0.9218, 0.9230, 0.9287, 0.9277, 0.9279, 0.9268, 0.9267]

# 创建绘图
fig, ax = plt.subplots(figsize=(10, 7))

# 绘制柱状图
# bars = ax.bar(disjoint_ratios, roc_auc, width=0.05, alpha=1, color="#F3D266", label='ROC AUC', zorder=2.5)
bars = ax.bar(disjoint_ratios, roc_auc, width=0.05, alpha=1, color=plt.get_cmap('Set1')(1), label='ROC AUC', zorder=2.5)
# 在每个柱子上方显示数值，增大字体大小
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), ha='center', va='bottom', fontsize=15)

ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=1.0, zorder=2)

# 设置字体大小
plt.xticks(disjoint_ratios, fontsize=15, fontweight='bold')
plt.yticks(fontsize=15, fontweight='bold')

# 设置坐标轴刻度和标签
plt.xlabel("Disjoint Ratios", fontsize=20)
plt.ylabel("AUC-ROC", fontsize=20)

# # 设置图例
# plt.legend(loc='best', fontsize=15)

# 设置坐标轴范围
plt.ylim(0.90, 0.935)

# # 设置标题
plt.title('Different Disjoint Ratios AUC-ROC Performance', fontsize=20, fontweight='bold')

# 显示图形
plt.show()
plt.show()