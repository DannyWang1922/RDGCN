import matplotlib.pyplot as plt
import numpy as np

# Aggregate categories
categories = ['mean', 'sum', 'max']

# Evaluation metrics for each category
# 将每个指标的值提取到对应的列表中
accuracy = [0.8471454880294658, 0.6808471454880296, 0.7767955801104972]
precision = [0.8736855889313409, 0.8758703201444374, 0.7880323865180567]
recall = [0.8117863720073665, 0.4386740331491713, 0.761694290976059]
f1_score = [0.8411392289024839, 0.5649995768063857, 0.7726545729098534]
roc_auc = [0.9278023666351253, 0.8572598177372146, 0.8410081092355748]
pr_auc = [0.9287087304359737, 0.8352449909245371, 0.7838911379287624]


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
ax.set_title('Different Aggregate Functions Performance (100 Epoch)', fontsize=20)

# Setting the x-axis tick marks to correspond to the aggregate categories
ax.set_xticks(indices + barWidth * 2.5)
ax.set_xticklabels(categories, fontsize=15, fontweight='bold')
plt.yticks(fontsize=15, fontweight='bold')

# Adding legend
ax.legend(loc='lower right')

# Show the plot
plt.show()