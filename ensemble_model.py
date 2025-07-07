#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
会話活性度予測のためのアンサンブル学習モデル
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib
matplotlib.use('Agg')  # バックエンドをAggに設定（Docker環境用）
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import DataProcessor


class EnsembleModel:
    """
    会話活性度予測のためのアンサンブル学習モデル
    """

    def __init__(self):
        """
        アンサンブルモデルの初期化
        """
        # 基本分類器の定義
        self.base_classifiers = {
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        
        # アンサンブルモデル
        self.ensemble_model = None
        self.best_model = None
        
        # 評価結果
        self.evaluation_results = {}

    def create_ensemble_model(self, voting='soft', weights=None):
        """
        アンサンブルモデルの作成
        :param voting: 投票方式 ('hard', 'soft')
        :param weights: 各分類器の重み
        """
        estimators = [(name, clf) for name, clf in self.base_classifiers.items()]
        
        if weights and len(weights) == len(estimators):
            self.ensemble_model = VotingClassifier(
                estimators=estimators,
                voting=voting,
                weights=weights
            )
        else:
            self.ensemble_model = VotingClassifier(
                estimators=estimators,
                voting=voting
            )
        
        print(f"アンサンブルモデル作成完了: {voting} voting")

    def train_model(self, X_train, y_train, optimize_hyperparams=False):
        """
        モデルの訓練
        :param X_train: 訓練データの説明変数
        :param y_train: 訓練データの目的変数
        :param optimize_hyperparams: ハイパーパラメータ最適化を行うかどうか
        """
        if optimize_hyperparams:
            self._optimize_hyperparameters(X_train, y_train)
        
        # モデルの訓練
        self.ensemble_model.fit(X_train, y_train)
        print("モデル訓練完了")

    def _optimize_hyperparameters(self, X_train, y_train):
        """
        ハイパーパラメータの最適化
        :param X_train: 訓練データの説明変数
        :param y_train: 訓練データの目的変数
        """
        # Random Forestのハイパーパラメータ最適化
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        rf_grid_search.fit(X_train, y_train)
        
        # 最適化されたパラメータで分類器を更新
        self.base_classifiers['random_forest'] = rf_grid_search.best_estimator_
        
        print(f"Random Forest最適化完了: {rf_grid_search.best_params_}")

    def evaluate_model(self, X_test, y_test):
        """
        モデルの評価
        :param X_test: テストデータの説明変数
        :param y_test: テストデータの目的変数
        """
        # 予測
        y_pred = self.ensemble_model.predict(X_test)
        y_pred_proba = self.ensemble_model.predict_proba(X_test)
        
        # 評価指標の計算
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)
        
        # 結果の保存
        self.evaluation_results = {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        print(f"モデル評価完了: 精度 {accuracy:.3f}")
        print("\n分類レポート:")
        print(classification_rep)
        
        return self.evaluation_results

    def cross_validate_model(self, X, y, cv=5):
        """
        クロスバリデーション
        :param X: データの説明変数
        :param y: データの目的変数
        :param cv: クロスバリデーション分割数
        """
        cv_scores = cross_val_score(self.ensemble_model, X, y, cv=cv, scoring='accuracy')
        
        print(f"クロスバリデーション結果:")
        print(f"平均精度: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return cv_scores

    def save_model(self, file_path):
        """
        モデルの保存
        :param file_path: 保存先ファイルパス
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.ensemble_model, f)
        print(f"モデルを保存: {file_path}")

    def load_model(self, file_path):
        """
        モデルの読み込み
        :param file_path: モデルファイルのパス
        """
        with open(file_path, 'rb') as f:
            self.ensemble_model = pickle.load(f)
        print(f"モデルを読み込み: {file_path}")

    def predict(self, X):
        """
        予測の実行
        :param X: 予測対象のデータ
        :return: 予測結果
        """
        if self.ensemble_model is None:
            raise ValueError("モデルが訓練されていません")
        
        return self.ensemble_model.predict(X)

    def predict_proba(self, X):
        """
        予測確率の取得
        :param X: 予測対象のデータ
        :return: 予測確率
        """
        if self.ensemble_model is None:
            raise ValueError("モデルが訓練されていません")
        
        return self.ensemble_model.predict_proba(X)

    def plot_confusion_matrix(self, save_path=None):
        """
        混同行列の可視化
        :param save_path: 保存先パス（指定した場合）
        """
        if 'confusion_matrix' not in self.evaluation_results:
            print("評価結果がありません")
            return
        
        cm = self.evaluation_results['confusion_matrix']
        class_names = ['LL', 'LM', 'MM', 'MH', 'HH']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('混同行列')
        plt.ylabel('実際のクラス')
        plt.xlabel('予測クラス')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混同行列を保存: {save_path}")
        
        plt.close()  # メモリを解放

    def get_feature_importance(self, feature_names):
        """
        特徴量重要度の取得（Random Forest使用時）
        :param feature_names: 特徴量名のリスト
        :return: 特徴量重要度のDataFrame
        """
        # VotingClassifierのfit済みRandomForestを取得
        rf_model = None
        for (name, _), clf in zip(self.ensemble_model.estimators, self.ensemble_model.estimators_):
            if name == 'random_forest':
                rf_model = clf
                break
        if rf_model is None:
            print("Random Forestがアンサンブルに含まれていません")
            return pd.DataFrame()
        
        importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df

    def plot_feature_importance(self, feature_names, save_path=None):
        """
        特徴量重要度の可視化（Random Forest使用時）
        :param feature_names: 特徴量名のリスト
        :param save_path: 保存先パス（指定した場合）
        """
        # VotingClassifierのfit済みRandomForestを取得
        rf_model = None
        for (name, _), clf in zip(self.ensemble_model.estimators, self.ensemble_model.estimators_):
            if name == 'random_forest':
                rf_model = clf
                break
        if rf_model is None:
            print("Random Forestがアンサンブルに含まれていません")
            return
        importances = rf_model.feature_importances_
        
        # 特徴量重要度のソート
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('特徴量重要度')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特徴量重要度を保存: {save_path}")
        
        plt.close()  # メモリを解放

    def train_with_optimized_params(self, X_train, y_train, rf_params=None, svm_params=None):
        """
        最適化されたパラメータでモデルを訓練
        :param X_train: 訓練データ
        :param y_train: 訓練ラベル
        :param rf_params: Random Forestの最適化パラメータ
        :param svm_params: SVMの最適化パラメータ
        """
        # デフォルトパラメータ
        default_rf_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        default_svm_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
        
        # 最適化パラメータで更新
        if rf_params:
            print(f"Random Forestパラメータ: {rf_params}")
            default_rf_params.update(rf_params)
        if svm_params:
            print(f"SVMパラメータ: {svm_params}")
            default_svm_params.update(svm_params)
        
        # モデル作成
        model1 = GaussianNB()
        model2 = KNeighborsClassifier(n_neighbors=5)
        model3 = RandomForestClassifier(**default_rf_params)
        # model4 = GradientBoostingClassifier(random_state=42)
        model4 = GradientBoostingClassifier()
        model5 = SVC(**default_svm_params)
        
        models = [
            ('naive_bayes', model1), 
            ('knn', model2), 
            ('random_forest', model3), 
            ('gradient_boosting', model4), 
            ('svm', model5)
        ]
        
        self.ensemble_model = VotingClassifier(estimators=models, voting='soft')
        
        # 訓練
        self.ensemble_model.fit(X_train, y_train)
        
        print("最適化パラメータでアンサンブルモデル訓練完了")
        print(f"Random Forest パラメータ: {default_rf_params}")
        print(f"SVM パラメータ: {default_svm_params}")


def create_binary_classifiers(data_processor, model_save_dir='./models'):
    """
    2クラス分類器の作成（参考プログラムの機能を再現）
    :param data_processor: データ処理クラスのインスタンス
    :param model_save_dir: モデル保存ディレクトリ
    """
    # クラスの組み合わせ
    class_combinations = [
        ['LL', 'LM'], ['LL', 'MM'], ['LL', 'MH'], ['LL', 'HH'],
        ['LM', 'MM'], ['LM', 'MH'], ['LM', 'HH'],
        ['MM', 'MH'], ['MM', 'HH'],
        ['MH', 'HH']
    ]
    
    # 保存ディレクトリの作成
    os.makedirs(model_save_dir, exist_ok=True)
    
    for class1, class2 in class_combinations:
        print(f"\n{class1} vs {class2} の分類器を作成中...")
        
        # 指定クラスのデータを抽出
        data_processor.get_target_classes(class1, class2)
        
        # 前処理
        data_processor.handle_missing_values()
        data_processor.standardize_data()
        data_processor.select_features(method='percentile', percentile=50)
        
        # データ分割
        X_train, X_test, y_train, y_test = data_processor.split_data()
        
        # アンサンブルモデルの作成と訓練
        ensemble = EnsembleModel()
        ensemble.create_ensemble_model(voting='soft')
        ensemble.train_model(X_train, y_train)
        
        # 評価
        results = ensemble.evaluate_model(X_test, y_test)
        
        # モデルの保存
        model_path = f"{model_save_dir}/ensemble_model_{class1}_{class2}.pickle"
        ensemble.save_model(model_path)
        
        print(f"分類器保存完了: {model_path}")


def main():
    """
    メイン関数
    """
    # データ処理
    processor = DataProcessor('./data/combined_features_with_activity.csv')
    
    # 全クラスのデータを使用
    processor.get_target_classes()
    
    # 前処理
    processor.handle_missing_values()
    processor.standardize_data()
    processor.select_features(method='percentile', percentile=50)
    
    # データ分割
    X_train, X_test, y_train, y_test = processor.split_data()
    
    # アンサンブルモデルの作成と訓練
    ensemble = EnsembleModel()
    ensemble.create_ensemble_model(voting='soft')
    ensemble.train_model(X_train, y_train)
    
    # 評価
    results = ensemble.evaluate_model(X_test, y_test)
    
    # クロスバリデーション
    X, y = processor.get_processed_data()
    cv_scores = ensemble.cross_validate_model(X, y)
    
    # モデルの保存
    ensemble.save_model('./models/ensemble_model_5class.pickle')
    
    # 2クラス分類器の作成
    create_binary_classifiers(processor)
    
    print("全ての処理が完了しました")


if __name__ == '__main__':
    main() 