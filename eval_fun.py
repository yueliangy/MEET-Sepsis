from scipy.optimize import linear_sum_assignment
import numpy as np


def calculate_acc(y_true, y_pred):
    """计算准确率（需要标签对齐）"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    return cost_matrix[row_ind, col_ind].sum() / y_true.size

def calculate_recall(y_true, y_pred):
    """计算召回率"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    recalls = []
    for i in range(len(row_ind)):
        tp = cost_matrix[row_ind[i], col_ind[i]]
        fn = np.sum(cost_matrix[row_ind[i], :]) - tp
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    return np.mean(recalls)

def calculate_f1(y_true, y_pred):
    """计算F1分数"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    f1_scores = []
    for i in range(len(row_ind)):
        tp = cost_matrix[row_ind[i], col_ind[i]]
        fp = np.sum(cost_matrix[:, col_ind[i]]) - tp
        fn = np.sum(cost_matrix[row_ind[i], :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if (precision + recall) > 0:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return np.mean(f1_scores)

def calculate_precision(y_true, y_pred):
    # 确保输入是 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 获取标签的唯一值
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    # 创建混淆矩阵
    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    # 计算每个簇的精确率
    precisions = []
    for j in range(len(col_ind)):
        tp = cost_matrix[row_ind[j], col_ind[j]]  # 簇中真实正确分类的样本数
        fp = np.sum(cost_matrix[:, col_ind[j]]) - tp  # 簇中真实错误分类的样本数
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
    # 计算聚类的整体精确率（按簇大小加权平均）
    precision = np.mean(precisions)
    return precision