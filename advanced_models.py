#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoostとディープラーニングモデル
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')  # バックエンドをAggに設定（Docker環境用）
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoostが利用できません")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlowが利用できません")


class XGBoostModel:
    """
    XGBoostモデル
    """

    def __init__(self, **params):
        """
        XGBoostモデルの初期化
        :param params: XGBoostのパラメータ
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoostがインストールされていません")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False
        }
        default_params.update(params)
        
        self.model = xgb.XGBClassifier(**default_params)
        self.evaluation_results = {}

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        モデルの訓練
        """
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        print("XGBoostモデル訓練完了")

    def predict(self, X):
        """
        予測
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        予測確率
        """
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        モデルの評価
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)
        
        self.evaluation_results = {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        print(f"XGBoost評価完了: 精度 {accuracy:.3f}")
        return self.evaluation_results

    def get_feature_importance(self, feature_names):
        """
        特徴量重要度の取得
        """
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df

    def save_model(self, file_path):
        """
        モデルの保存
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"XGBoostモデルを保存: {file_path}")

    def load_model(self, file_path):
        """
        モデルの読み込み
        """
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"XGBoostモデルを読み込み: {file_path}")


class DeepLearningModel:
    """
    ディープラーニングモデル
    """

    def __init__(self, input_dim, num_classes=5, **params):
        """
        ディープラーニングモデルの初期化
        :param input_dim: 入力特徴量の次元数
        :param num_classes: クラス数
        :param params: モデルパラメータ
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlowがインストールされていません")
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.evaluation_results = {}
        
        # デフォルトパラメータ
        self.params = {
            'hidden_layers': [64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10
        }
        self.params.update(params)
        
        self._build_model()

    def _build_model(self):
        """
        モデルの構築
        """
        self.model = Sequential()
        
        # 入力層
        self.model.add(Dense(
            self.params['hidden_layers'][0], 
            activation='relu', 
            input_shape=(self.input_dim,)
        ))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.params['dropout_rate']))
        
        # 隠れ層
        for units in self.params['hidden_layers'][1:]:
            self.model.add(Dense(units, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.params['dropout_rate']))
        
        # 出力層
        self.model.add(Dense(self.num_classes, activation='softmax'))
        
        # コンパイル
        self.model.compile(
            optimizer=Adam(learning_rate=self.params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("ディープラーニングモデル構築完了")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        モデルの訓練
        """
        callbacks = []
        if X_val is not None and y_val is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.params['patience'],
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        print("ディープラーニングモデル訓練完了")

    def predict(self, X):
        """
        予測
        """
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1) + 1  # 1-5のクラスに変換

    def predict_proba(self, X):
        """
        予測確率
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        モデルの評価
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)
        
        self.evaluation_results = {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        print(f"ディープラーニング評価完了: 精度 {accuracy:.3f}")
        return self.evaluation_results

    def plot_training_history(self, save_path=None):
        """
        訓練履歴の可視化
        """
        if self.history is None:
            print("訓練履歴がありません")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 損失
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 精度
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            # PDF形式で保存
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            print(f"訓練履歴を保存: {pdf_path}")
        
        plt.close()  # メモリを解放

    def save_model(self, file_path):
        """
        モデルの保存
        """
        self.model.save(file_path)
        print(f"ディープラーニングモデルを保存: {file_path}")

    def load_model(self, file_path):
        """
        モデルの読み込み
        """
        self.model = tf.keras.models.load_model(file_path)
        print(f"ディープラーニングモデルを読み込み: {file_path}")


def compare_models(models_dict, X_test, y_test):
    """
    複数モデルの比較
    :param models_dict: モデルの辞書 {'model_name': model}
    :param X_test: テストデータ
    :param y_test: テストラベル
    """
    results = {}
    
    print("=" * 60)
    print("モデル比較結果")
    print("=" * 60)
    
    for name, model in models_dict.items():
        print(f"\n{name}の評価:")
        print("-" * 30)
        
        # 予測
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # 評価指標
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_rep
        }
        
        print(f"精度: {accuracy:.3f}")
        print("分類レポート:")
        print(classification_rep)
    
    # 結果の比較
    print("\n" + "=" * 60)
    print("精度比較")
    print("=" * 60)
    
    for name, result in results.items():
        print(f"{name}: {result['accuracy']:.3f}")
    
    # 最良モデルの特定
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n最良モデル: {best_model[0]} (精度: {best_model[1]['accuracy']:.3f})")
    
    return results


class AdvancedModels:
    """
    高度なモデル（XGBoost、ディープラーニング）の管理クラス
    """
    
    def __init__(self):
        """
        初期化
        """
        self.xgb_model = None
        self.dl_model = None
        self.models = {}
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, **params):
        """
        XGBoostモデルの訓練
        
        Args:
            X_train: 訓練データの特徴量
            y_train: 訓練データの目的変数
            X_test: テストデータの特徴量
            y_test: テストデータの目的変数
            **params: XGBoostのパラメータ
            
        Returns:
            訓練されたXGBoostモデル
        """
        if not XGBOOST_AVAILABLE:
            print("XGBoostが利用できません")
            return None
        
        try:
            # ラベルエンコーディング（0始まりに変換）
            y_train_encoded = y_train - 1
            y_test_encoded = y_test - 1
            
            self.xgb_model = XGBoostModel(**params)
            self.xgb_model.train(X_train, y_train_encoded, X_test, y_test_encoded)
            self.xgb_model.evaluate(X_test, y_test_encoded)
            
            self.models['xgboost'] = self.xgb_model
            return self.xgb_model
            
        except Exception as e:
            print(f"XGBoost訓練エラー: {e}")
            return None
    
    def train_deep_learning(self, X_train, y_train, X_test, y_test, **params):
        """
        ディープラーニングモデルの訓練
        
        Args:
            X_train: 訓練データの特徴量
            y_train: 訓練データの目的変数
            X_test: テストデータの特徴量
            y_test: テストデータの目的変数
            **params: ディープラーニングのパラメータ
            
        Returns:
            訓練されたディープラーニングモデル
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlowが利用できません")
            return None
        
        try:
            # ラベルエンコーディング（0始まりに変換）
            y_train_encoded = y_train - 1
            y_test_encoded = y_test - 1
            
            input_dim = X_train.shape[1]
            num_classes = len(np.unique(y_train))
            
            self.dl_model = DeepLearningModel(input_dim, num_classes, **params)
            self.dl_model.train(X_train, y_train_encoded, X_test, y_test_encoded)
            self.dl_model.evaluate(X_test, y_test_encoded)
            
            self.models['deep_learning'] = self.dl_model
            return self.dl_model
            
        except Exception as e:
            print(f"ディープラーニング訓練エラー: {e}")
            return None
    
    def predict(self, model_name, X):
        """
        指定したモデルで予測
        
        Args:
            model_name: モデル名 ('xgboost' or 'deep_learning')
            X: 予測対象データ
            
        Returns:
            予測結果
        """
        if model_name not in self.models:
            raise ValueError(f"モデル '{model_name}' が見つかりません")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        
        # ラベルを元に戻す（+1）
        if model_name in ['xgboost', 'deep_learning']:
            predictions = predictions + 1
        
        return predictions
    
    def get_model(self, model_name):
        """
        指定したモデルを取得
        
        Args:
            model_name: モデル名
            
        Returns:
            モデルオブジェクト
        """
        return self.models.get(model_name)
    
    def save_models(self, models_dir):
        """
        全モデルを保存
        
        Args:
            models_dir: 保存先ディレクトリ
        """
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if hasattr(model, 'save_model'):
                file_path = os.path.join(models_dir, f"{model_name}_model.pickle")
                model.save_model(file_path)
    
    def load_models(self, models_dir):
        """
        全モデルを読み込み
        
        Args:
            models_dir: 読み込み元ディレクトリ
        """
        for model_name in ['xgboost', 'deep_learning']:
            file_path = os.path.join(models_dir, f"{model_name}_model.pickle")
            if os.path.exists(file_path):
                if model_name == 'xgboost' and XGBOOST_AVAILABLE:
                    model = XGBoostModel()
                    model.load_model(file_path)
                    self.models[model_name] = model
                elif model_name == 'deep_learning' and TENSORFLOW_AVAILABLE:
                    input_dim = 10  # デフォルト値、実際の値に合わせて調整
                    model = DeepLearningModel(input_dim)
                    model.load_model(file_path)
                    self.models[model_name] = model


if __name__ == '__main__':
    # テスト用
    print("XGBoost利用可能:", XGBOOST_AVAILABLE)
    print("TensorFlow利用可能:", TENSORFLOW_AVAILABLE) 