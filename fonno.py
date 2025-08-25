import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# 数据字典
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
time_steps = ['4h', '6h', '10h', '16h']
methods = list(data['4h'].keys())
metrics = ['Accuracy', 'Precision', 'Recall', 'DCV', 'F1-score']
markers = ['o', 's', '^', 'D']
colors = ['red', 'orange', 'blue', 'green']

# 对每个指标画一张图
for metric_index, metric_name in enumerate(metrics):
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        y_values = [np.mean(data[t][method][metric_index]) for t in time_steps]
        plt.plot(time_steps, y_values, marker=markers[i], color=colors[i],
                 label=method, linewidth=2)

    plt.xlabel('预测时间步长', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'{metric_name} 在不同时间步长下的表现', fontsize=14)
    plt.ylim(0.4, 0.85)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(f'{metric_name}_comparison_plot.svg')
    plt.show()
