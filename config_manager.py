#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
設定管理クラス
YAML設定ファイルを読み込み、データ処理と機械学習の設定を管理する
"""

import os
import yaml
import pandas as pd
from typing import Dict, Any, List, Optional, Union


class ConfigManager:
    """
    設定管理クラス
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス（Noneの場合はデフォルト設定）
        """
        if config_path and os.path.exists(config_path):
            self.config = self._load_config(config_path)
        else:
            self.config = self._get_default_config()
        
        self._validate_config()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        設定ファイルを読み込み
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            設定辞書
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"設定ファイルを読み込み: {config_path}")
            return config
        except Exception as e:
            print(f"設定ファイルの読み込みエラー: {e}")
            print("デフォルト設定を使用します")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        デフォルト設定を取得
        
        Returns:
            デフォルト設定辞書
        """
        return {
            'data': {
                'columns': {
                    'conversation_id': 'cid',
                    'target': 'class',
                    'exclude': []
                },
                'classes': {
                    'numeric': {1: 'LL', 2: 'LM', 3: 'MM', 4: 'MH', 5: 'HH'},
                    'string': {'LL': 1, 'LM': 2, 'MM': 3, 'MH': 4, 'HH': 5}
                },
                'validation': {
                    'min_samples': 10,
                    'min_features': 3,
                    'max_missing_ratio': 0.5
                }
            },
            'preprocessing': {
                'missing_values': {
                    'strategy': 'mean',
                    'numeric_strategy': 'mean',
                    'categorical_strategy': 'mode'
                },
                'scaling': {
                    'enabled': True,
                    'method': 'standard',
                    'by_conversation': True
                },
                'discretization': {
                    'enabled': True,
                    'n_bins': 20,
                    'method': 'uniform'
                },
                'feature_selection': {
                    'enabled': True,
                    'method': 'percentile',
                    'percentile': 50,
                    'k_best': 10
                },
                'temporal_features': {
                    'enabled': False,
                    'window_size': 3
                }
            },
            'split': {
                'test_size': 0.2,
                'random_state': 42,
                'by_conversation': True,
                'stratify': True
            },
            'models': {
                'ensemble': {
                    'enabled': True,
                    'voting': 'soft',
                    'classifiers': ['rf', 'svm', 'nb']
                },
                'xgboost': {
                    'enabled': True,
                    'use_label_encoding': True
                },
                'deep_learning': {
                    'enabled': True,
                    'architecture': [128, 64, 32],
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001,
                    'epochs': 50,
                    'patience': 10
                }
            },
            'optimization': {
                'enabled': True,
                'method': 'random',
                'n_trials': 50,
                'cv_folds': 5
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1'],
                'cross_validation': True,
                'cv_folds': 5
            },
            'output': {
                'models_dir': './models',
                'plots_dir': './models',
                'save_format': 'pdf',
                'save_plots': True,
                'save_models': True
            }
        }
    
    def _validate_config(self):
        """
        設定の妥当性を検証
        """
        # 必須キーのチェック
        required_keys = ['data', 'preprocessing', 'split', 'models', 'output']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"必須設定キー '{key}' が見つかりません")
    
    def get_data_config(self) -> Dict[str, Any]:
        """データ設定を取得"""
        return self.config.get('data', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """前処理設定を取得"""
        return self.config.get('preprocessing', {})
    
    def get_split_config(self) -> Dict[str, Any]:
        """データ分割設定を取得"""
        return self.config.get('split', {})
    
    def get_models_config(self) -> Dict[str, Any]:
        """モデル設定を取得"""
        return self.config.get('models', {})
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """最適化設定を取得"""
        return self.config.get('optimization', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """評価設定を取得"""
        return self.config.get('evaluation', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """出力設定を取得"""
        return self.config.get('output', {})
    
    def analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        データ構造を分析して設定を自動調整
        
        Args:
            df: 分析対象のデータフレーム
            
        Returns:
            分析結果辞書
        """
        analysis = {
            'columns': list(df.columns),
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'conversation_id_col': None,
            'target_col': None,
            'feature_cols': [],
            'class_type': None,
            'class_values': None
        }
        
        # 会話IDカラムの検出
        conv_id_candidates = ['cid', 'conversation_id', 'conv_id', 'session_id']
        for col in conv_id_candidates:
            if col in df.columns:
                analysis['conversation_id_col'] = col
                break
        
        # 目的変数カラムの検出
        target_candidates = ['class', 'target', 'label', 'activity_level', 'impression']
        for col in target_candidates:
            if col in df.columns:
                analysis['target_col'] = col
                break
        
        # 特徴量カラムの特定
        exclude_cols = [analysis['conversation_id_col'], analysis['target_col']]
        analysis['feature_cols'] = [col for col in df.columns if col not in exclude_cols and col is not None]
        
        # クラス形式の検出
        if analysis['target_col']:
            target_col = analysis['target_col']
            unique_values = df[target_col].dropna().unique()
            analysis['class_values'] = sorted(unique_values)
            
            # 数値クラスか文字列クラスかを判定
            if pd.api.types.is_numeric_dtype(df[target_col]):
                analysis['class_type'] = 'numeric'
            else:
                analysis['class_type'] = 'string'
        
        return analysis
    
    def auto_adjust_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        データ分析結果に基づいて設定を自動調整
        
        Args:
            analysis: データ分析結果
            
        Returns:
            調整された設定辞書
        """
        adjusted_config = self.config.copy()
        
        # カラム設定の調整
        if analysis['conversation_id_col']:
            adjusted_config['data']['columns']['conversation_id'] = analysis['conversation_id_col']
        
        if analysis['target_col']:
            adjusted_config['data']['columns']['target'] = analysis['target_col']
        
        # クラス設定の調整
        if analysis['class_type'] == 'numeric':
            # 数値クラスの場合、自動的にマッピングを作成
            class_mapping = {}
            for i, val in enumerate(analysis['class_values'], 1):
                class_mapping[val] = f"Class_{i}"
            adjusted_config['data']['classes']['numeric'] = class_mapping
        
        # データ検証設定の調整
        n_samples, n_features = analysis['shape']
        adjusted_config['data']['validation']['min_samples'] = min(10, n_samples // 5)
        adjusted_config['data']['validation']['min_features'] = min(3, n_features // 3)
        
        # 会話単位処理の有効/無効
        if analysis['conversation_id_col']:
            adjusted_config['preprocessing']['scaling']['by_conversation'] = True
            adjusted_config['split']['by_conversation'] = True
        else:
            adjusted_config['preprocessing']['scaling']['by_conversation'] = False
            adjusted_config['split']['by_conversation'] = False
        
        return adjusted_config
    
    def print_config_summary(self):
        """
        設定のサマリーを表示
        """
        print("=" * 50)
        print("設定サマリー")
        print("=" * 50)
        
        # データ設定
        data_config = self.get_data_config()
        print(f"目的変数: {data_config['columns']['target']}")
        if data_config['columns']['conversation_id']:
            print(f"会話ID: {data_config['columns']['conversation_id']}")
        
        # 前処理設定
        preproc_config = self.get_preprocessing_config()
        print(f"標準化: {'有効' if preproc_config['scaling']['enabled'] else '無効'}")
        print(f"離散化: {'有効' if preproc_config['discretization']['enabled'] else '無効'}")
        print(f"特徴量選択: {'有効' if preproc_config['feature_selection']['enabled'] else '無効'}")
        
        # モデル設定
        models_config = self.get_models_config()
        print(f"アンサンブル: {'有効' if models_config['ensemble']['enabled'] else '無効'}")
        print(f"XGBoost: {'有効' if models_config['xgboost']['enabled'] else '無効'}")
        print(f"ディープラーニング: {'有効' if models_config['deep_learning']['enabled'] else '無効'}")
        
        print("=" * 50) 