#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ハイパーパラメータ最適化
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer
import matplotlib
matplotlib.use('Agg')  # バックエンドをAggに設定（Docker環境用）
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterOptimizer:
    """
    ハイパーパラメータ最適化クラス
    """

    def __init__(self):
        """
        初期化
        """
        self.best_params = {}
        self.best_scores = {}
        self.optimization_results = {}

    def optimize_random_forest(self, X_train, y_train, cv=5, method='grid'):
        """
        Random Forestのハイパーパラメータ最適化
        """
        print("Random Forestのハイパーパラメータ最適化開始")
        
        # パラメータグリッド
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # ランダム検索用のパラメータ分布
        param_distributions = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [None, 5, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        if method == 'grid':
            search = GridSearchCV(
                rf, param_grid, cv=cv, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
        else:  # random
            search = RandomizedSearchCV(
                rf, param_distributions, n_iter=50, cv=cv, 
                scoring='accuracy', n_jobs=-1, verbose=1, random_state=42
            )
        
        search.fit(X_train, y_train)
        
        self.best_params['random_forest'] = search.best_params_
        self.best_scores['random_forest'] = search.best_score_
        self.optimization_results['random_forest'] = search
        
        print(f"Random Forest最適化完了: 最良精度 {search.best_score_:.3f}")
        print(f"最良パラメータ: {search.best_params_}")
        
        return search.best_estimator_

    def optimize_svm(self, X_train, y_train, cv=5, method='grid'):
        """
        SVMのハイパーパラメータ最適化
        """
        print("SVMのハイパーパラメータ最適化開始")
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        svm = SVC(probability=True, random_state=42)
        
        search = GridSearchCV(
            svm, param_grid, cv=cv, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        search.fit(X_train, y_train)
        
        self.best_params['svm'] = search.best_params_
        self.best_scores['svm'] = search.best_score_
        self.optimization_results['svm'] = search
        
        print(f"SVM最適化完了: 最良精度 {search.best_score_:.3f}")
        print(f"最良パラメータ: {search.best_params_}")
        
        return search.best_estimator_

    def optimize_xgboost(self, X_train, y_train, cv=5, method='optuna'):
        """
        XGBoostのハイパーパラメータ最適化
        """
        if not XGBOOST_AVAILABLE:
            print("XGBoostが利用できません")
            return None
        
        print("XGBoostのハイパーパラメータ最適化開始")
        
        if method == 'optuna' and OPTUNA_AVAILABLE:
            return self._optimize_xgboost_optuna(X_train, y_train, cv)
        else:
            return self._optimize_xgboost_grid(X_train, y_train, cv)

    def _optimize_xgboost_grid(self, X_train, y_train, cv=5):
        """
        XGBoostのグリッド検索最適化
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        
        search = GridSearchCV(
            xgb_model, param_grid, cv=cv, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        search.fit(X_train, y_train)
        
        self.best_params['xgboost'] = search.best_params_
        self.best_scores['xgboost'] = search.best_score_
        self.optimization_results['xgboost'] = search
        
        print(f"XGBoost最適化完了: 最良精度 {search.best_score_:.3f}")
        print(f"最良パラメータ: {search.best_params_}")
        
        return search.best_estimator_

    def _optimize_xgboost_optuna(self, X_train, y_train, cv=5, n_trials=100):
        """
        Optunaを使用したXGBoost最適化
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }
            
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['xgboost'] = study.best_params
        self.best_scores['xgboost'] = study.best_value
        
        print(f"XGBoost最適化完了: 最良精度 {study.best_value:.3f}")
        print(f"最良パラメータ: {study.best_params}")
        
        # 最良パラメータでモデルを構築
        best_model = xgb.XGBClassifier(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        
        return best_model

    def optimize_ensemble_weights(self, base_models, X_train, y_train, cv=5):
        """
        アンサンブル重みの最適化
        """
        print("アンサンブル重みの最適化開始")
        
        from sklearn.ensemble import VotingClassifier
        
        # 重みの候補
        weight_combinations = [
            [1, 1, 1, 1, 1],  # 均等重み
            [2, 1, 1, 1, 1],  # Random Forest重視
            [1, 2, 1, 1, 1],  # SVM重視
            [1, 1, 2, 1, 1],  # XGBoost重視
            [1, 1, 1, 2, 1],  # KNN重視
            [1, 1, 1, 1, 2],  # Naive Bayes重視
        ]
        
        best_score = 0
        best_weights = None
        
        for weights in weight_combinations:
            ensemble = VotingClassifier(
                estimators=[
                    ('rf', base_models['random_forest']),
                    ('svm', base_models['svm']),
                    ('xgb', base_models['xgboost']),
                    ('knn', base_models['knn']),
                    ('nb', base_models['naive_bayes'])
                ],
                voting='soft',
                weights=weights
            )
            
            scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy')
            mean_score = scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                best_weights = weights
        
        self.best_params['ensemble_weights'] = best_weights
        self.best_scores['ensemble'] = best_score
        
        print(f"アンサンブル重み最適化完了: 最良精度 {best_score:.3f}")
        print(f"最良重み: {best_weights}")
        
        return best_weights

    def plot_optimization_results(self, save_path=None):
        """
        最適化結果の可視化
        """
        if not self.optimization_results:
            print("最適化結果がありません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (model_name, search) in enumerate(self.optimization_results.items()):
            if i >= 4:
                break
                
            if hasattr(search, 'cv_results_'):
                # グリッド検索結果の可視化
                results_df = pd.DataFrame(search.cv_results_)
                
                # 重要なパラメータを選択
                param_cols = [col for col in results_df.columns if col.startswith('param_')]
                if len(param_cols) > 2:
                    param_cols = param_cols[:2]  # 最初の2つのパラメータのみ
                
                if len(param_cols) == 2:
                    pivot_table = results_df.pivot_table(
                        values='mean_test_score',
                        index=param_cols[0],
                        columns=param_cols[1],
                        aggfunc='mean'
                    )
                    
                    sns.heatmap(pivot_table, annot=True, fmt='.3f', ax=axes[i])
                    axes[i].set_title(f'{model_name} Optimization Results')
        
        plt.tight_layout()
        
        if save_path:
            # PDF形式で保存
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            print(f"最適化結果を保存: {pdf_path}")
        
        plt.close()  # メモリを解放

    def get_best_models(self):
        """
        最適化されたモデルの取得
        """
        return self.best_params, self.best_scores

    def save_optimization_results(self, file_path):
        """
        最適化結果の保存
        """
        import pickle
        
        results = {
            'best_params': self.best_params,
            'best_scores': self.best_scores,
            'optimization_results': self.optimization_results
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"最適化結果を保存: {file_path}")


def cross_val_score(estimator, X, y, cv=5, scoring='accuracy'):
    """
    クロスバリデーションスコアの計算
    """
    from sklearn.model_selection import cross_val_score as cv_score
    return cv_score(estimator, X, y, cv=cv, scoring=scoring)


if __name__ == '__main__':
    print("XGBoost利用可能:", XGBOOST_AVAILABLE)
    print("Optuna利用可能:", OPTUNA_AVAILABLE) 