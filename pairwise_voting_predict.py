#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ペアワイズ学習済みモデルを使って活性度を予測し、各クラス確率も出力するスクリプト
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. コマンドライン引数
parser = argparse.ArgumentParser(description='ペアワイズモデルによる活性度予測スクリプト')
parser.add_argument('--data', '-d', type=str, required=True, help='予測対象データcsv（class列含む）')
parser.add_argument('--model-dir', '-m', type=str, required=True, help='モデルディレクトリ（ensemble）')
parser.add_argument('--output', '-o', type=str, default='pairwise_predict_result.csv', help='出力csv')
args = parser.parse_args()

# 2. データ読み込み
print(f"データ読み込み: {args.data}")
df = pd.read_csv(args.data)

# クラスラベルとクラス名の対応
class_labels = [1, 2, 3, 4, 5]
class_names = ['LL', 'LM', 'MM', 'MH', 'HH']
label_to_name = dict(zip(class_labels, class_names))
pair_list = []
for i in range(len(class_labels)):
    for j in range(i+1, len(class_labels)):
        pair_list.append([class_labels[i], class_labels[j]])

# 3. 特徴量・標準化
X = df.drop(['class', 'cid'], axis=1, errors='ignore')
y_true = df['class']
cid = df['cid'] if 'cid' in df.columns else np.arange(len(df))
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# 4. 各ペアのモデルで確率予測
# 各サンプルごとにクラスごとの勝者確率リストを作る
probs = {k: np.zeros(len(df)) for k in class_labels}
counts = {k: 0 for k in class_labels}

for label1, label2 in pair_list:
    name1, name2 = label_to_name[label1], label_to_name[label2]
    model_path = os.path.join(args.model_dir, f"{name1}_vs_{name2}.pickle")
    if not os.path.exists(model_path):
        print(f"モデルが見つかりません: {model_path}")
        continue
    with open(model_path, 'rb') as fp:
        d = pickle.load(fp)
        if isinstance(d, dict) and 'model' in d and 'features' in d:
            model = d['model']
            features = d['features']
        else:
            model = d
            features = X_scaled.columns.tolist()  # 古い形式は全カラム
    # 特徴量を揃える
    X_pred = X_scaled[features]
    prob = model.predict_proba(X_pred)
    # model.classes_の順番で [0,1] = [label1, label2] になるはず
    probs[label1] += prob[:,0]
    probs[label2] += prob[:,1]
    counts[label1] += 1
    counts[label2] += 1

# 5. 各クラスの平均確率を計算
prob_matrix = np.zeros((len(df), len(class_labels)))
for idx, k in enumerate(class_labels):
    if counts[k] > 0:
        prob_matrix[:,idx] = probs[k] / counts[k]
    else:
        prob_matrix[:,idx] = 0
# 各サンプルごとに確率を正規化（合計1になるように）
row_sums = prob_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # 0割防止
prob_matrix = prob_matrix / row_sums

# 6. 最終予測（最大確率のクラス）
pred_class = np.argmax(prob_matrix, axis=1) + 1  # 1始まり

# 7. 結果をDataFrameにまとめて保存
result_df = pd.DataFrame({
    'cid': cid,
    'true_class': y_true,
    'pred_class': pred_class
})
for idx, k in enumerate(class_labels):
    result_df[f'prob_{k}'] = prob_matrix[:,idx]

result_df.to_csv(args.output, index=False)
print(f"予測結果を保存: {args.output}") 