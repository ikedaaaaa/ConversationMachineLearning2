#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
柔軟な会話活性度予測システム
設定ファイルに基づいて様々なデータ形式に対応
"""

import os
import sys
import argparse
import datetime
import warnings
from typing import Optional

# 設定とデータ処理
from config_manager import ConfigManager
from flexible_data_processor import FlexibleDataProcessor

# モデル関連
from ensemble_model import EnsembleModel
from advanced_models import AdvancedModels
from hyperparameter_optimization import HyperparameterOptimizer

# 警告を抑制
warnings.filterwarnings('ignore')


def create_sample_data(file_path: str, n_samples: int = 100, n_features: int = 16):
    """
    サンプルデータを作成
    
    Args:
        file_path: 保存先ファイルパス
        n_samples: サンプル数
        n_features: 特徴量数
    """
    import pandas as pd
    import numpy as np
    
    # より現実的なデータ分布
    n_conversations = max(20, n_samples // 10)  # 最低20会話
    samples_per_conv = n_samples // n_conversations
    
    conversation_ids = []
    feature_data = []
    classes = []
    
    # 各会話でデータを生成
    for conv_id in range(1, n_conversations + 1):
        conv_samples = samples_per_conv
        
        # 会話ごとに特徴的なパターンを生成
        base_features = np.random.randn(n_features) * 0.5  # 会話固有のベース
        conv_class_bias = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.2, 0.3, 0.2, 0.15])  # 会話の傾向
        
        for _ in range(conv_samples):
            # ノイズを加えた特徴量
            features = base_features + np.random.randn(n_features) * 0.3
            
            # クラスを生成（会話の傾向に基づく）
            if np.random.random() < 0.7:  # 70%は会話の傾向に従う
                class_val = conv_class_bias
            else:  # 30%はランダム
                class_val = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.2, 0.3, 0.2, 0.15])
            
            conversation_ids.append(conv_id)
            feature_data.append(features)
            classes.append(class_val)
    
    # データフレームを作成
    feature_cols = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(feature_data, columns=feature_cols)
    df['cid'] = conversation_ids
    df['class'] = classes
    
    # 保存
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"サンプルデータを作成: {file_path}")
    print(f"  サンプル数: {len(df)}")
    print(f"  会話数: {df['cid'].nunique()}")
    print(f"  クラス分布: {df['class'].value_counts().sort_index().to_dict()}")


def main():
    """
    メイン処理
    """
    parser = argparse.ArgumentParser(description='柔軟な会話活性度予測システム')
    parser.add_argument('--data', type=str, default=None, help='データファイルのパス')
    parser.add_argument('--config', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--create-sample', action='store_true', help='サンプルデータを作成')
    parser.add_argument('--sample-size', type=int, default=100, help='サンプルデータのサイズ')
    parser.add_argument('--sample-features', type=int, default=16, help='サンプルデータの特徴量数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("柔軟な会話活性度予測システム")
    print("=" * 60)
    print(f"開始時刻: {datetime.datetime.now()}")
    
    # サンプルデータ作成
    if args.create_sample:
        sample_file = f"./data/sample_data_{args.sample_size}samples_{args.sample_features}features.csv"
        create_sample_data(sample_file, args.sample_size, args.sample_features)
        args.data = sample_file
    
    # データファイルの指定がない場合
    if args.data is None:
        print("\n使用方法:")
        print("  1. サンプルデータで実行:")
        print("     python main_flexible.py --create-sample")
        print("  2. 既存データで実行:")
        print("     python main_flexible.py --data ./data/your_data.csv")
        print("  3. カスタム設定で実行:")
        print("     python main_flexible.py --data ./data/your_data.csv --config custom_config.yaml")
        print("\n利用可能なサンプルデータ:")
        print("  - ./data/combined_features_with_activity.csv")
        return
    
    try:
        # 1. 設定管理
        print("\n1. 設定管理")
        print("-" * 30)
        config_manager = ConfigManager(args.config)
        config_manager.print_config_summary()
        
        # 2. データ処理
        print("\n2. データ処理")
        print("-" * 30)
        data_processor = FlexibleDataProcessor(config_manager)
        
        # データ処理実行
        X_train, X_test, y_train, y_test = data_processor.process_data(args.data)
        
        # 処理結果の表示
        processed_data = data_processor.get_processed_data()
        print(f"処理済みデータ:")
        print(f"  訓練データ: {X_train.shape}")
        print(f"  テストデータ: {X_test.shape}")
        print(f"  特徴量名: {processed_data['feature_names']}")
        
        # 3. モデル学習
        print("\n3. モデル学習")
        print("-" * 30)
        
        # アンサンブルモデル
        models_config = config_manager.get_models_config()
        if models_config['ensemble']['enabled']:
            print("アンサンブルモデルを学習中...")
            ensemble_model = EnsembleModel()
            ensemble_model.create_ensemble_model(voting='soft')
            ensemble_model.train_model(X_train, y_train)
            
            # モデル保存
            output_config = config_manager.get_output_config()
            models_dir = output_config['models_dir']
            os.makedirs(models_dir, exist_ok=True)
            ensemble_model.save_model(os.path.join(models_dir, 'ensemble_model.pickle'))
            
            # 評価
            ensemble_model.evaluate_model(X_test, y_test)
            
            # 可視化
            if output_config['save_plots']:
                plots_dir = output_config['plots_dir']
                ensemble_model.plot_confusion_matrix(os.path.join(plots_dir, 'confusion_matrix.pdf'))
                ensemble_model.plot_feature_importance(processed_data['feature_names'], os.path.join(plots_dir, 'feature_importance.pdf'))
        
        # 高度なモデル
        advanced_models = AdvancedModels()
        
        # XGBoost
        if models_config['xgboost']['enabled']:
            try:
                print("XGBoostモデルを学習中...")
                xgb_model = advanced_models.train_xgboost(X_train, y_train, X_test, y_test)
                if xgb_model:
                    print("XGBoost学習完了")
            except Exception as e:
                print(f"XGBoost学習エラー: {e}")
        
        # ディープラーニング
        if models_config['deep_learning']['enabled']:
            try:
                print("ディープラーニングモデルを学習中...")
                dl_model = advanced_models.train_deep_learning(X_train, y_train, X_test, y_test)
                if dl_model:
                    print("ディープラーニング学習完了")
            except Exception as e:
                print(f"ディープラーニング学習エラー: {e}")
        
        # 4. ハイパーパラメータ最適化
        optimization_config = config_manager.get_optimization_config()
        if optimization_config['enabled']:
            print("\n4. ハイパーパラメータ最適化")
            print("-" * 30)
            
            try:
                optimizer = HyperparameterOptimizer()
                best_params = optimizer.optimize_ensemble(X_train, y_train)
                print(f"最適パラメータ: {best_params}")
            except Exception as e:
                print(f"ハイパーパラメータ最適化エラー: {e}")
        
        # 5. 予測テスト
        print("\n5. 予測テスト")
        print("-" * 30)
        
        # テストデータの一部で予測
        test_samples = min(5, len(X_test))
        X_test_sample = X_test.iloc[:test_samples]
        y_test_sample = y_test.iloc[:test_samples]
        
        print(f"テストサンプル {test_samples} 個で予測:")
        
        # アンサンブルモデルの予測
        if models_config['ensemble']['enabled']:
            ensemble_predictions = ensemble_model.predict(X_test_sample)
            for i, (true, pred) in enumerate(zip(y_test_sample, ensemble_predictions)):
                print(f"  サンプル {i+1}: 実際={true}, 予測={pred}")
        
        # 6. 処理結果のサマリー
        print("\n6. 処理結果のサマリー")
        print("-" * 30)
        
        print(f"データサンプル数: {len(X_train) + len(X_test)}")
        print(f"特徴量数: {len(processed_data['feature_names'])}")
        print(f"訓練データ数: {len(X_train)}")
        print(f"テストデータ数: {len(X_test)}")
        
        if models_config['ensemble']['enabled']:
            print(f"アンサンブルモデル: 学習完了")
        
        print(f"終了時刻: {datetime.datetime.now()}")
        print("=" * 60)
        print("処理が完了しました！")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 