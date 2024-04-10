import matplotlib.pyplot as plt
import numpy as np

# Aggregate categories
categories = ['mean', 'sum', 'max']

# Evaluation metrics for each category
accuracy = [0.8449355432780846, 0.7640883977900552, 0.6629834254143646]
precision = [0.88295509573987, 0.7541041512624341, 0.7430135251877743]
recall = [0.7955801104972375, 0.7900552486187845, 0.5141804788213629]
f1_score = [0.8366682700679478, 0.7699281896539134, 0.5982856634517948]
roc_auc = [0.9286957052593022, 0.8247835332661803, 0.7464478427941081]
pr_auc = [0.929205819287232, 0.7915288341144995, 0.7309339768863071]

# Number of categories
n_categories = len(categories)

# The x position of bars
barWidth = 0.1
indices = np.arange(n_categories)

# Set up the matplotlib figure and axes
fig, ax = plt.subplots(figsize=(10, 7))

ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=1.0, zorder=2)

# 获取预设的颜色方案
color_cmap = plt.get_cmap('Set1')

# Creating bars for each metric
ax.bar(indices, accuracy, width=barWidth, color=color_cmap(0), edgecolor='grey', label='Accuracy', zorder=2.5)
ax.bar(indices + barWidth, precision, width=barWidth, color=color_cmap(1), edgecolor='grey', label='Precision', zorder=2.5)
ax.bar(indices + barWidth * 2, recall, width=barWidth, color=color_cmap(2), edgecolor='grey', label='Recall', zorder=2.5)
ax.bar(indices + barWidth * 3, f1_score, width=barWidth, color=color_cmap(3), edgecolor='grey', label='F1 Score', zorder=2.5)
ax.bar(indices + barWidth * 4, roc_auc, width=barWidth, color=color_cmap(4), edgecolor='grey', label='ROC AUC', zorder=2.5)
ax.bar(indices + barWidth * 5, pr_auc, width=barWidth, color=color_cmap(5), edgecolor='grey', label='PR AUC', zorder=2.5)

# Adding labels and title
ax.set_xlabel('Aggregation Function Type', fontsize=20)
ax.set_ylabel('Metric Value', fontsize=20)
ax.set_title('Different Aggregate Functions Performance (100 Epoch)', fontsize=20, fontweight='bold')

# Setting the x-axis tick marks to correspond to the aggregate categories
ax.set_xticks(indices + barWidth * 2.5)
ax.set_xticklabels(categories, fontsize=15, fontweight='bold')
plt.yticks(fontsize=15, fontweight='bold')

# Adding legend
ax.legend(loc='lower right')

# Show the plot
plt.show()