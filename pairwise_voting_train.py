#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2クラスペアごとにバイナリ分類器（アンサンブル＋ディープラーニング）を学習・保存するスクリプト
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# 必要ならディープラーニング用のimport
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# 1. コマンドライン引数
parser = argparse.ArgumentParser(description='2クラスペアごとバイナリ分類学習スクリプト')
parser.add_argument('--data', '-d', type=str, required=True, help='学習データファイルのパス')
parser.add_argument('--feature-selection', '-fs', type=str, default=None, help='特徴量選択方法（指定しない場合は自動選択）')
parser.add_argument('--use-all-data', '-all', action='store_true', help='全データを訓練データとして使う')
parser.add_argument('--output-dir', '-o', type=str, default='models', help='モデル保存先ディレクトリ')
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

# 3. 各ペアで処理
for label1, label2 in pair_list:
    name1, name2 = label_to_name[label1], label_to_name[label2]
    print(f"\n=== {name1}({label1}) vs {name2}({label2}) ===")
    # 3-1. 対象2クラスのデータのみ抽出
    df_pair = df[df['class'].isin([label1, label2])].copy()
    if len(df_pair) == 0:
        print(f"データがありません: {label1}, {label2}")
        continue
    X = df_pair.drop(['class', 'cid'], axis=1, errors='ignore')
    y = df_pair['class'].map({label1: 0, label2: 1})  # 0/1に変換

    # 4. 欠損値処理（会話単位で平均補完）
    if 'cid' in df_pair.columns:
        X = X.copy()
        for cid, group in df_pair.groupby('cid'):
            idx = group.index
            X.loc[idx] = X.loc[idx].fillna(X.loc[idx].mean())
    else:
        X = X.fillna(X.mean())

    # 5. 標準化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # 6. 特徴量選択
    if args.feature_selection:
        print(f"特徴量選択方法: {args.feature_selection}")
        if args.feature_selection == 'percentile':
            selector = SelectPercentile(score_func=f_classif, percentile=50)
            X_selected = selector.fit_transform(X_scaled, y)
            selected_cols = X_scaled.columns[selector.get_support()]
            X_selected = pd.DataFrame(X_selected, columns=selected_cols, index=X_scaled.index)
        elif args.feature_selection == 'k_best':
            selector = SelectKBest(score_func=f_classif, k=10)
            X_selected = selector.fit_transform(X_scaled, y)
            selected_cols = X_scaled.columns[selector.get_support()]
            X_selected = pd.DataFrame(X_selected, columns=selected_cols, index=X_scaled.index)
        elif args.feature_selection == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=10)
            X_selected = selector.fit_transform(X_scaled, y)
            selected_cols = X_scaled.columns[selector.get_support()]
            X_selected = pd.DataFrame(X_selected, columns=selected_cols, index=X_scaled.index)
        elif args.feature_selection == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=10)
            X_selected = selector.fit_transform(X_scaled, y)
            selected_cols = X_scaled.columns[selector.get_support()]
            X_selected = pd.DataFrame(X_selected, columns=selected_cols, index=X_scaled.index)
        elif args.feature_selection == 'lasso':
            estimator = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)
            selector = SelectFromModel(estimator)
            X_selected = selector.fit_transform(X_scaled, y)
            selected_cols = X_scaled.columns[selector.get_support()]
            X_selected = pd.DataFrame(X_selected, columns=selected_cols, index=X_scaled.index)
        else:
            print(f"未対応の特徴量選択方法: {args.feature_selection}。全特徴量を使用します。")
            X_selected = X_scaled.copy()
    else:
        # 複数の方法で比較
        print("特徴量選択方法を自動比較中...")
        methods = [
            ('percentile', SelectPercentile(score_func=f_classif, percentile=50)),
            ('k_best', SelectKBest(score_func=f_classif, k=10)),
            ('mutual_info', SelectKBest(score_func=mutual_info_classif, k=10)),
            ('rfe', RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=10)),
            ('lasso', SelectFromModel(LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)))
        ]
        from sklearn.model_selection import cross_val_score
        best_score = -1
        best_method = None
        best_selector = None
        best_selected_cols = None
        for name, selector in methods:
            try:
                X_tmp = selector.fit_transform(X_scaled, y)
                if hasattr(selector, 'get_support'):
                    selected_cols = X_scaled.columns[selector.get_support()]
                    X_tmp = pd.DataFrame(X_tmp, columns=selected_cols, index=X_scaled.index)
                else:
                    selected_cols = X_scaled.columns
                # 簡易なモデルでCV
                model = RandomForestClassifier(n_estimators=30, random_state=42)
                scores = cross_val_score(model, X_tmp, y, cv=3, scoring='accuracy')
                mean_score = scores.mean()
                print(f"  {name}: CV精度={mean_score:.3f} (特徴量数: {X_tmp.shape[1]})")
                if mean_score > best_score:
                    best_score = mean_score
                    best_method = name
                    best_selector = selector
                    best_selected_cols = selected_cols
            except Exception as e:
                print(f"  {name}: エラー {e}")
        print(f"最良の特徴量選択方法: {best_method} (CV精度={best_score:.3f})")
        # 最良の方法で特徴量選択
        X_selected = best_selector.transform(X_scaled)
        X_selected = pd.DataFrame(X_selected, columns=best_selected_cols, index=X_scaled.index)

    # 7. データ分割
    X_test = None
    y_test = None
    if args.use_all_data:
        X_train, X_test, y_train, y_test = X_selected, None, y, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)

    # 8. ハイパーパラメータ最適化
    print("ハイパーパラメータ最適化中...")
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 3, 5]}
    knn_params = {'n_neighbors': [3, 5, 7]}
    rf = RandomForestClassifier(random_state=42)
    knn = KNeighborsClassifier()
    gnb = GaussianNB()
    # RandomForest最適化
    rf_gs = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1)
    rf_gs.fit(X_train, y_train)
    print(f"RF最適パラメータ: {rf_gs.best_params_}")
    # KNN最適化
    knn_gs = GridSearchCV(knn, knn_params, cv=3, n_jobs=-1)
    knn_gs.fit(X_train, y_train)
    print(f"KNN最適パラメータ: {knn_gs.best_params_}")
    # アンサンブル
    ensemble = VotingClassifier(estimators=[('rf', rf_gs.best_estimator_), ('knn', knn_gs.best_estimator_), ('gnb', gnb)], voting='soft')
    ensemble.fit(X_train, y_train)

    # 10. ディープラーニングモデル
    if TENSORFLOW_AVAILABLE:
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    else:
        model = None

    # 11. 評価・保存
    # 保存ディレクトリ構造: models/<csvファイル名>/<アンサンブル or DL>/<name1>_vs_<name2>.<拡張子>
    csv_base = os.path.splitext(os.path.basename(args.data))[0]
    ens_dir = os.path.join(args.output_dir, csv_base, 'ensemble')
    dl_dir = os.path.join(args.output_dir, csv_base, 'dl')
    os.makedirs(ens_dir, exist_ok=True)
    os.makedirs(dl_dir, exist_ok=True)
    # アンサンブル保存
    with open(os.path.join(ens_dir, f"{name1}_vs_{name2}.pickle"), 'wb') as fp:
        pickle.dump({'model': ensemble, 'features': X_selected.columns.tolist()}, fp)
    # DL保存
    if model is not None:
        model.save(os.path.join(dl_dir, f"{name1}_vs_{name2}.h5"))
    print(f"モデル保存: {ens_dir}, {dl_dir}")

    # 評価
    print(f"[DEBUG] args.use_all_data={args.use_all_data}, type(X_test)={type(X_test)}, X_test is None={X_test is None}, len(X_test)={len(X_test) if X_test is not None else 'None'}")
    print(f"[DEBUG] type(y_test)={type(y_test)}, y_test is None={y_test is None}, len(y_test)={len(y_test) if y_test is not None else 'None'}")
    if not args.use_all_data and X_test is not None:
        y_pred = ensemble.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"アンサンブル正解率: {acc:.3f}")
        if model is not None:
            y_pred_dl = (model.predict(X_test) > 0.5).astype(int).flatten()
            acc_dl = accuracy_score(y_test, y_pred_dl)
            print(f"DL正解率: {acc_dl:.3f}")
        print(classification_report(y_test, y_pred, target_names=[name1, name2])) 