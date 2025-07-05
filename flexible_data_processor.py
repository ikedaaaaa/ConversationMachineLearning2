#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
柔軟なデータ処理クラス
設定ファイルに基づいて様々なデータ形式に対応する
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectPercentile, SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings

from config_manager import ConfigManager


class FlexibleDataProcessor:
    """
    柔軟なデータ処理クラス
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        初期化
        
        Args:
            config_manager: 設定管理オブジェクト
        """
        self.config_manager = config_manager
        self.df = None
        self.analysis = None
        self.processed_data = None
        self.scalers = {}
        self.feature_selector = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        データを読み込み
        
        Args:
            file_path: データファイルのパス
            
        Returns:
            読み込んだデータフレーム
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")
        
        # ファイル拡張子に基づいて読み込み
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            self.df = pd.read_excel(file_path)
        elif file_path.endswith('.parquet'):
            self.df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"サポートされていないファイル形式: {file_path}")
        
        print(f"データ読み込み完了: {self.df.shape[0]} サンプル, {self.df.shape[1]} 特徴量")
        return self.df
    
    def analyze_data(self) -> Dict[str, Any]:
        """
        データ構造を分析
        
        Returns:
            分析結果辞書
        """
        if self.df is None:
            raise ValueError("データが読み込まれていません")
        
        self.analysis = self.config_manager.analyze_data_structure(self.df)
        
        print("データ構造分析結果:")
        print(f"  カラム数: {len(self.analysis['columns'])}")
        print(f"  サンプル数: {self.analysis['shape'][0]}")
        print(f"  特徴量数: {len(self.analysis['feature_cols'])}")
        print(f"  目的変数: {self.analysis['target_col']}")
        print(f"  クラス形式: {self.analysis['class_type']}")
        if self.analysis['class_values']:
            print(f"  クラス値: {self.analysis['class_values']}")
        if self.analysis['conversation_id_col']:
            print(f"  会話ID: {self.analysis['conversation_id_col']}")
        
        return self.analysis
    
    def validate_data(self) -> bool:
        """
        データの妥当性を検証
        
        Returns:
            妥当性チェック結果
        """
        if self.df is None or self.analysis is None:
            raise ValueError("データが分析されていません")
        
        validation_config = self.config_manager.get_data_config()['validation']
        
        # サンプル数チェック
        if self.df.shape[0] < validation_config['min_samples']:
            print(f"警告: サンプル数が少なすぎます ({self.df.shape[0]} < {validation_config['min_samples']})")
            return False
        
        # 特徴量数チェック
        if len(self.analysis['feature_cols']) < validation_config['min_features']:
            print(f"警告: 特徴量数が少なすぎます ({len(self.analysis['feature_cols'])} < {validation_config['min_features']})")
            return False
        
        # 欠損値チェック
        missing_ratio = self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])
        if missing_ratio > validation_config['max_missing_ratio']:
            print(f"警告: 欠損値が多すぎます ({missing_ratio:.2%} > {validation_config['max_missing_ratio']:.2%})")
            return False
        
        print("データ妥当性チェック完了")
        return True
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        欠損値処理
        
        Returns:
            処理後のデータフレーム
        """
        if self.df is None:
            raise ValueError("データが読み込まれていません")
        
        missing_config = self.config_manager.get_preprocessing_config()['missing_values']
        
        # 数値カラムとカテゴリカラムを分離
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        # 数値カラムの欠損値処理
        if missing_config['numeric_strategy'] == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif missing_config['numeric_strategy'] == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif missing_config['numeric_strategy'] == 'drop':
            self.df = self.df.dropna(subset=numeric_cols)
        
        # カテゴリカラムの欠損値処理
        if missing_config['categorical_strategy'] == 'mode':
            for col in categorical_cols:
                mode_val = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col] = self.df[col].fillna(mode_val)
        elif missing_config['categorical_strategy'] == 'drop':
            self.df = self.df.dropna(subset=categorical_cols)
        
        print("欠損値処理完了")
        return self.df
    
    def encode_classes(self) -> pd.DataFrame:
        """
        クラスエンコーディング
        
        Returns:
            エンコーディング後のデータフレーム
        """
        if self.df is None or self.analysis is None:
            raise ValueError("データが分析されていません")
        
        target_col = self.analysis['target_col']
        if target_col is None:
            return self.df
        
        class_config = self.config_manager.get_data_config()['classes']
        
        if self.analysis['class_type'] == 'string':
            # 文字列クラスを数値に変換
            string_mapping = class_config['string']
            self.df[target_col] = self.df[target_col].map(string_mapping)
            print(f"文字列クラスを数値に変換: {string_mapping}")
        elif self.analysis['class_type'] == 'numeric':
            # 数値クラスを正規化（必要に応じて）
            unique_values = sorted(self.df[target_col].unique())
            if unique_values != list(range(1, len(unique_values) + 1)):
                # 連続していない場合は正規化
                value_mapping = {val: i+1 for i, val in enumerate(unique_values)}
                self.df[target_col] = self.df[target_col].map(value_mapping)
                print(f"数値クラスを正規化: {value_mapping}")
        
        return self.df
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        特徴量のスケーリング
        
        Args:
            X: 特徴量データフレーム
            fit: フィッティングを行うかどうか
            
        Returns:
            スケーリング後のデータフレーム
        """
        scaling_config = self.config_manager.get_preprocessing_config()['scaling']
        
        if not scaling_config['enabled']:
            return X
        
        # スケーラーを選択
        if scaling_config['method'] == 'standard':
            scaler = StandardScaler()
        elif scaling_config['method'] == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_config['method'] == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"サポートされていないスケーリング方法: {scaling_config['method']}")
        
        # 会話単位でのスケーリング
        if scaling_config['by_conversation'] and self.analysis['conversation_id_col']:
            conv_id_col = self.analysis['conversation_id_col']
            scaled_data = []
            
            for conv_id in X[conv_id_col].unique():
                conv_mask = X[conv_id_col] == conv_id
                conv_data = X[conv_mask].drop(columns=[conv_id_col])
                
                if fit:
                    conv_scaled = scaler.fit_transform(conv_data)
                    self.scalers[conv_id] = scaler
                else:
                    conv_scaled = self.scalers[conv_id].transform(conv_data)
                
                conv_scaled_df = pd.DataFrame(conv_scaled, columns=conv_data.columns, index=conv_data.index)
                conv_scaled_df[conv_id_col] = conv_id
                scaled_data.append(conv_scaled_df)
            
            result = pd.concat(scaled_data).sort_index()
        else:
            # 全体でのスケーリング
            if fit:
                scaled_data = scaler.fit_transform(X)
                self.scalers['global'] = scaler
            else:
                scaled_data = self.scalers['global'].transform(X)
            
            result = pd.DataFrame(scaled_data, columns=X.columns, index=X.index)
        
        print(f"スケーリング完了: {scaling_config['method']}")
        return result
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, fit: bool = True) -> pd.DataFrame:
        """
        特徴量選択
        
        Args:
            X: 特徴量データフレーム
            y: 目的変数
            fit: フィッティングを行うかどうか
            
        Returns:
            選択後のデータフレーム
        """
        selection_config = self.config_manager.get_preprocessing_config()['feature_selection']
        
        if not selection_config['enabled']:
            return X
        
        # 特徴量選択方法を選択
        if selection_config['method'] == 'percentile':
            selector = SelectPercentile(percentile=selection_config['percentile'])
        elif selection_config['method'] == 'k_best':
            selector = SelectKBest(k=selection_config['k_best'])
        elif selection_config['method'] == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=selection_config['k_best'])
        else:
            raise ValueError(f"サポートされていない特徴量選択方法: {selection_config['method']}")
        
        if fit:
            X_selected = selector.fit_transform(X, y)
            self.feature_selector = selector
            selected_features = X.columns[selector.get_support()].tolist()
            print(f"特徴量選択完了: {len(selected_features)} 特徴量選択")
        else:
            X_selected = self.feature_selector.transform(X)
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        データ分割
        
        Args:
            X: 特徴量データフレーム
            y: 目的変数
            
        Returns:
            訓練・テストデータのタプル
        """
        split_config = self.config_manager.get_split_config()
        
        # 会話単位での分割
        if split_config['by_conversation'] and self.analysis['conversation_id_col']:
            conv_id_col = self.analysis['conversation_id_col']
            unique_conversations = X[conv_id_col].unique()
            
            # 会話を訓練・テストに分割
            train_convs, test_convs = train_test_split(
                unique_conversations,
                test_size=split_config['test_size'],
                random_state=split_config['random_state'],
                stratify=None  # 会話単位では層別サンプリングは困難
            )
            
            # 会話IDに基づいてデータを分割
            train_mask = X[conv_id_col].isin(train_convs)
            test_mask = X[conv_id_col].isin(test_convs)
            
            X_train = X[train_mask].drop(columns=[conv_id_col])
            X_test = X[test_mask].drop(columns=[conv_id_col])
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            print(f"会話単位でデータ分割: 訓練会話 {len(train_convs)}, テスト会話 {len(test_convs)}")
        else:
            # 通常の分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=split_config['test_size'],
                random_state=split_config['random_state'],
                stratify=y if split_config['stratify'] else None
            )
        
        print(f"データ分割完了: 訓練データ {len(X_train)}, テストデータ {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def process_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        データ処理の全工程を実行
        
        Args:
            file_path: データファイルのパス
            
        Returns:
            処理済みデータのタプル
        """
        # 1. データ読み込み
        self.load_data(file_path)
        
        # 2. データ分析
        self.analyze_data()
        
        # 3. データ妥当性チェック
        if not self.validate_data():
            warnings.warn("データ妥当性チェックで警告が発生しました")
        
        # 4. 欠損値処理
        self.handle_missing_values()
        
        # 5. クラスエンコーディング
        self.encode_classes()
        
        # 6. 特徴量と目的変数を分離
        target_col = self.analysis['target_col']
        feature_cols = self.analysis['feature_cols']
        conv_id_col = self.analysis['conversation_id_col']
        
        X = self.df[feature_cols].copy()
        y = self.df[target_col]
        
        # 会話IDカラムを追加（存在する場合）
        if conv_id_col:
            X[conv_id_col] = self.df[conv_id_col]
        
        # 7. スケーリング
        X = self.scale_features(X, fit=True)
        
        # 8. 特徴量選択
        X = self.select_features(X, y, fit=True)
        
        # 9. データ分割
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        self.processed_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist()
        }
        
        return X_train, X_test, y_train, y_test
    
    def get_processed_data(self) -> Dict[str, Any]:
        """
        処理済みデータを取得
        
        Returns:
            処理済みデータ辞書
        """
        if self.processed_data is None:
            raise ValueError("データが処理されていません")
        return self.processed_data 