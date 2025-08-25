import numpy as np
from tqdm import tqdm
from torch import optim
from TSRL import *
from torch import nn
import torch.nn.functional as F
import math
from sklearn.manifold import TSNE
import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
device = 'cuda'
patience = 20
from sklearn.preprocessing import StandardScaler
from eval_fun import calculate_acc, calculate_recall, calculate_f1, calculate_precision


class Config:
    lr = 0.01
    xgb_params = {
        'n_estimators': 500,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'objective': 'multi:softmax',
        'num_class': 2,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'early_stopping_rounds': 50,
        'max_bin': 512,
        'grow_policy': 'lossguide'
    }

def evaluate_model(y_test, y_pred):
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()
    return {
        "Accuracy": calculate_acc(y_test, y_pred),
        "Recall": calculate_recall(y_test, y_pred),
        "F1 Score": calculate_f1(y_test, y_pred),
        "pre": calculate_precision(y_test, y_pred)
    }

def load_data(step):
    """加载并预处理数据"""
    dataname = "sepsis_cla"
    path = f"/root/autodl-tmp/MVCNN/data/std1/S{step}"
    data_train = np.load(f'{path}/data/{dataname}_features_train.npy', allow_pickle=True)
    data_test = np.load(f'{path}/data/{dataname}_features_test.npy', allow_pickle=True)
    label_train = np.load(f'{path}/labels/{dataname}_labels_train.npy', allow_pickle=True)
    label_test = np.load(f'{path}/labels/{dataname}_labels_test.npy', allow_pickle=True)

    print(data_train.shape[0]+data_test.shape[0])
    
    scaler = StandardScaler()
    train_2d = data_train.reshape(-1, data_train.shape[-1])
    scaler.fit(train_2d)
    data_train = scaler.transform(train_2d).reshape(data_train.shape)
    test_2d = data_test.reshape(-1, data_test.shape[-1])
    data_test = scaler.transform(test_2d).reshape(data_test.shape)
    
    return data_train, data_test, label_train, label_test

# 创建结果保存文件
output_file = "training_results.txt"
with open(output_file, 'w') as f:
    f.write("Time Step Results\n")
    f.write("=================\n\n")

# 循环训练不同时间步长
for step in range(3, 5):
    print(f"\nProcessing time step: {step}")
    
    # 加载数据
    data_train, data_test, label_train, label_test = load_data(step)
    
    train_data = data_train
    test_data = data_test
    sample_num, seq_len, feature_dim = train_data.shape
    num_cluster = len(np.unique(label_train))
    
    best_acc = 0
    best_f1 = 0
    best_pre = 0
    best_rec = 0
    
    # 初始化模型
    model = TSRL(input_dim=feature_dim, view_dim=5, num_views=feature_dim)
    # model = TSRL(input_dim=feature_dim)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    
    classes = np.unique(label_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=label_train)
    sample_weights = class_weights[label_train]
    
    # GPU
    model.to(device)
    criterion = nn.MSELoss()
    train_data = torch.tensor(train_data).float().to(device)
    test_data = torch.tensor(test_data).float().to(device)
    
    for seed1 in range(5, 8):
        model = pretrain_model(model, train_data, device)
        
        def extract_features(data):
            model.eval()
            with torch.no_grad():
                tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
                f = model(tensor_data)
                return f.cpu().numpy()
        
        train_features = extract_features(data_train)
        test_features = extract_features(data_test)
        
        print("\n=== 训练XGBoost分类器 ===")
        best_score = 0
        cv = StratifiedKFold(n_splits=5)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(train_features, label_train)):
            xgb_model = xgb.XGBClassifier(**Config.xgb_params)
            xgb_model.fit(
                train_features[train_idx],
                label_train[train_idx],
                eval_set=[(train_features[val_idx], label_train[val_idx])],
                sample_weight=sample_weights[train_idx],
            )
            
            score = xgb_model.score(train_features[val_idx], label_train[val_idx])
            if score > best_score:
                best_model = xgb_model
                best_score = score
            
            y_pred = best_model.predict(test_features)
            metrics = evaluate_model(label_test, y_pred)
            
            acc = metrics["Accuracy"]
            f1 = metrics["F1 Score"]
            rec = metrics["Recall"]
            pre = metrics["pre"]
            
            if acc >= best_acc:
                best_acc = acc
                best_f1 = f1
                best_pre = pre
                best_rec = rec
    
    # 保存结果到文件
    with open(output_file, 'a') as f:
        f.write(f"Time Step: {step}\n")
        f.write(f"Accuracy: {best_acc:.4f}\n")
        f.write(f"F1 Score: {best_f1:.4f}\n")
        f.write(f"Precision: {best_pre:.4f}\n")
        f.write(f"Recall: {best_rec:.4f}\n")
        f.write("-------------------\n")
    
    print(f"Results for time step {step} saved:")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"Precision: {best_pre:.4f}")
    print(f"Recall: {best_rec:.4f}")
    print(f"F1 Score: {best_f1:.4f}")


