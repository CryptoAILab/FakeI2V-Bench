

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

np.random.seed(42)

# ----------- 所有检测器 -----------
detectors  = [
            'CNNDet', 'CoDE', 'DeFake',  'DMID', 'DRCT', 
            'DIRE',
            'LGrad', 
            'LNP', 
              'NPR', 'Patch', 'RINE', 
            'UniFD'
              ]

# ----------- 基本参数 -----------
base_dir = "/data/lp/KDD26-benchmark/VFTData"
train_csv_name = "split.csv"
feature_names = ["mean", "var", "max", "min", "median", "random"]

# ----------- 模型配置 -----------
MODEL_CONFIGS = {
    "RF": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [800],
            "max_depth": [4],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }
    )
}

# ----------- 视频级特征提取 -----------
def extract_video_features(csv_path):
    df = pd.read_csv(csv_path)
    video_groups = df.groupby("video_id")

    X, y = [], []
    for vid, group in video_groups:
        scores = group["y_prob"].values.astype(float)
        label = int(group["label"].iloc[0])
        features = [
            np.mean(scores),
            np.var(scores),
            np.max(scores),
            np.min(scores),
            np.median(scores),
            np.random.choice(scores)
        ]
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

# ----------- 模型训练 -----------
def train_model(model_name, X, y, cv=3):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model_name}")
    base_model, param_grid = MODEL_CONFIGS[model_name]

    if param_grid:
        search = GridSearchCV(base_model, param_grid, cv=cv, scoring="f1", n_jobs=-1)
        search.fit(X, y)
        best_model = search.best_estimator_
        print(f"{model_name} best params: {search.best_params_}")
    else:
        base_model.fit(X, y)
        best_model = base_model
    return best_model

# ----------- 阈值优化 -----------
def optimize_threshold(model, X, y, num_steps=1000):
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    thresholds = np.linspace(0, 1, num_steps)
    best_f1 = 0
    best_tau = 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = t
    return best_tau

# ----------- 测试评估 -----------
def evaluate_model(model, tau, csv_dir, scaler):
    results = {}
    for file in sorted(os.listdir(csv_dir)):
        if not file.endswith(".csv") or file == train_csv_name:
            continue
        X_test, y_test = extract_video_features(os.path.join(csv_dir, file))
        X_test = scaler.transform(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        preds = (probs >= tau).astype(int)
        auc = roc_auc_score(y_test, probs)
        ap = average_precision_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        results[file] = {"AUC": round(float(auc),6),
                         "AP": round(float(ap),6),
                         "F1": round(float(f1),6)}
    return results

# ----------- 主循环：所有检测器 -----------
if __name__ == "__main__":
    models_to_run = ["RF"]  # 可以改成你想跑的模型列表
    for detector in detectors:
        print(f"\n==== Processing detector: {detector} ====")
        csv_dir = os.path.join(base_dir, detector)
        train_csv = os.path.join(csv_dir, train_csv_name)
        X_train, y_train = extract_video_features(train_csv)

        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        for model_name in models_to_run:
            print(f"\nTraining model {model_name} on {detector} ...")
            model = train_model(model_name, X_train_scaled, y_train)
            tau = optimize_threshold(model, X_train_scaled, y_train)
            print(f"Optimal threshold (tau*): {tau:.3f}")
            results = evaluate_model(model, tau, csv_dir, scaler)

            # 保存 JSON
            output_file_name = f"{detector}_{model_name}_fusion_results.json"
            with open(output_file_name, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved: {output_file_name}")

    print("\nAll detectors processed.")
