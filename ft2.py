import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False

# 原始数据
data = {
    '16h': {
        'MAC-Boost(本作品)': [0.7857, 0.7980, 0.7852, 0.7939, 0.7742],
        'XGBoost': [0.7857, 0.7826, 0.7752, 0.8010, 0.7657],
        'RandomForest': [0.7253, 0.7258, 0.6950, 0.7519, 0.6918],
        'OneStepKNN': [0.4945, 0.5395, 0.5236, 0.4873, 0.4657]
    },
    '10h': {
        'MAC-Boost(本作品)': [0.7857, 0.7846, 0.7924, 0.7995, 0.7787],
        'XGBoost': [0.7582, 0.7500, 0.7412, 0.7807, 0.7349],
        'RandomForest': [0.7198, 0.7136, 0.6917, 0.7528, 0.6881],
        'OneStepKNN': [0.4780, 0.4950, 0.4963, 0.4768, 0.4371]
    },
    '6h': {
        'MAC-Boost(本作品)': [0.7912, 0.7874, 0.7979, 0.8034, 0.7826],
        'XGBoost': [0.7527, 0.7397, 0.7443, 0.7808, 0.7384],
        'RandomForest': [0.7253, 0.7240, 0.6972, 0.7533, 0.6954],
        'OneStepKNN': [0.4890, 0.4876, 0.5206, 0.5079, 0.4672]
    },
    '4h': {
        'MAC-Boost(本作品)': [0.7967, 0.7948, 0.8066, 0.8092, 0.7894],
        'XGBoost': [0.7527, 0.7432, 0.7475, 0.7788, 0.7394],
        'RandomForest': [0.7253, 0.7258, 0.6950, 0.7519, 0.6918],
        'OneStepKNN': [0.4835, 0.4550, 0.5138, 0.5324, 0.4579]
    }
}

metrics = ['Accuracy(准确率)', 'Precision(精确率)', 'Recall(召回率)', 'DCV(一致性)', 'F1-score(F1分数)']
methods = ['MAC-Boost(本作品)', 'XGBoost', 'RandomForest', 'OneStepKNN']
colors = ['#f94144', '#577590', '#90be6d', '#adb5bd']
bar_width = 0.2
x = np.arange(len(metrics))

# 分别绘制每个时间步一张图
for time in ['4h', '6h', '10h', '16h']:
    plt.figure(figsize=(8, 5))
    for i, method in enumerate(methods):
        scores = data[time][method]
        plt.bar(x + i * bar_width, scores, width=bar_width, color=colors[i], label=method)

    plt.xticks(x + 1.5 * bar_width, metrics, rotation=30)
    plt.ylim(0.4, 0.85)
    plt.ylabel('得分值（越接近1越好）')
    plt.title(f'预测时间步长为 {time} 时各算法的表现')
    plt.legend(loc='upper center', ncol=2, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'bar_plot_{time}.svg')
    plt.show()