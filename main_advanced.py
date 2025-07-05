#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
会話活性度予測システム - 高度版
時系列特徴量、XGBoost、ディープラーニング、ハイパーパラメータ最適化を含む
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')  # バックエンドをAggに設定（Docker環境用）
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# フォント設定（日本語フォントの警告を避けるため簡素化）
plt.rcParams['font.family'] = ['DejaVu Sans']

from data_processor import DataProcessor
from ensemble_model import EnsembleModel
from advanced_models import XGBoostModel, DeepLearningModel, compare_models
from hyperparameter_optimization import HyperparameterOptimizer

# ライブラリの利用可能性をチェック
try:
    import xgboost
    XGBOOST_AVAILABLE = True
    print("XGBoostが利用できます")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoostが利用できません")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    print("TensorFlowが利用できます")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlowが利用できません")


def create_sample_data_with_conversations():
    """
    会話IDを含むサンプルデータを作成
    """
    np.random.seed(42)
    
    # 会話IDと区間IDを生成
    conversations = []
    for conv_id in range(1, 31):  # 30会話に増加
        for interval_id in range(1, 6):  # 各会話5区間
            conversations.append(f"{conv_id}_{interval_id}")
    
    # 特徴量を生成（会話内で相関を持つように）
    data = []
    for conv_id in range(1, 31):  # 30会話に増加
        # 会話ごとのベース値を設定
        base_values = np.random.normal(0, 1, 16)
        
        for interval_id in range(1, 6):
            # 区間ごとに少し変化させる
            features = base_values + np.random.normal(0, 0.2, 16)
            
            # クラスを会話ごとに設定（より均等に分布）
            if conv_id <= 6:
                class_label = 1  # LL
            elif conv_id <= 12:
                class_label = 2  # LM
            elif conv_id <= 18:
                class_label = 3  # MM
            elif conv_id <= 24:
                class_label = 4  # MH
            else:
                class_label = 5  # HH
            
            # ランダムに少し変更
            if np.random.random() < 0.3:
                class_label = np.random.randint(1, 6)
            
            row = [f"{conv_id}_{interval_id}"] + list(features) + [class_label]
            data.append(row)
    
    # カラム名
    feature_names = [
        'speech_ratio', 'longest_speech_ratio', 'speech_speed_ratio',
        'silence_total', 'speaker_change_count', 'turn_taking_frequency',
        'response_time', 'interruption_count', 'overlap_ratio',
        'speech_volume', 'speech_clarity', 'engagement_level',
        'topic_consistency', 'conversation_flow', 'participant_balance',
        'overall_activity'
    ]
    
    columns = ['cid'] + feature_names + ['class']
    df = pd.DataFrame(data, columns=columns)
    
    # ファイル保存
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_data_with_conversations.csv', index=False)
    
    print(f"サンプルデータ作成完了: {len(df)} サンプル, {len(feature_names)} 特徴量")
    print(f"会話数: {len(df['cid'].str.split('_').str[0].unique())}")
    print(f"クラス分布: {df['class'].value_counts().sort_index().to_dict()}")
    
    return df


def main():
    """
    メイン処理
    """
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='会話活性度予測システム - 高度版')
    parser.add_argument('--data', '-d', type=str, 
                       help='学習データファイルのパス (指定しない場合はサンプルデータを生成)')
    parser.add_argument('--test', '-t', action='store_true',
                       help='予測テストを実行する')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("会話活性度予測システム - 高度版")
    print("=" * 60)
    print(f"開始時刻: {datetime.datetime.now()}")
    
    # データファイルの指定がある場合
    if args.data:
        if not os.path.exists(args.data):
            print(f"エラー: データファイルが見つかりません: {args.data}")
            print("使用例:")
            print("  python main_advanced.py --data ./data/your_data.csv")
            print("  python main_advanced.py -d ./data/your_data.csv")
            return
        
        print(f"\n指定されたデータファイル: {args.data}")
        data_file = args.data
        use_sample_data = False
    else:
        # 1. サンプルデータ作成
        print("\n1. サンプルデータ作成")
        print("-" * 30)
        df = create_sample_data_with_conversations()
        data_file = 'data/sample_data_with_conversations.csv'
        use_sample_data = True
    
    # 2. データ処理
    print("\n2. データ処理")
    print("-" * 30)
    processor = DataProcessor(data_file)
    
    # 時系列特徴量を追加（現在は無効化）
    processor.add_temporal_features()
    
    # 会話ごとの前処理
    processor.process_by_conversation()
    
    # 目的変数を設定
    processor.get_target_classes()
    
    # 会話単位での標準化
    processor.standardize_by_conversation()
    
    # 特徴量選択
    processor.select_features()
    
    # 会話単位でのデータ分割（テストサイズを調整）
    X_train, X_test, y_train, y_test, train_conv_ids, test_conv_ids = processor.split_by_conversation(test_size=0.3)
    
    # XGBoost用ラベル（0始まり）
    y_train_xgb = y_train - 1
    y_test_xgb = y_test - 1
    print(f"処理完了: 訓練データ {len(X_train)}, テストデータ {len(X_test)}")
    
    # 3. ハイパーパラメータ最適化
    print("\n3. ハイパーパラメータ最適化")
    print("-" * 30)
    optimizer = HyperparameterOptimizer()
    
    # Random Forest最適化
    best_rf = optimizer.optimize_random_forest(X_train, y_train, cv=5, method='random')
    
    # SVM最適化
    best_svm = optimizer.optimize_svm(X_train, y_train, cv=5)
    
    # XGBoost最適化（0始まりラベル）
    best_xgb = optimizer.optimize_xgboost(X_train, y_train_xgb, cv=5, method='grid')
    
    # 4. モデル訓練
    print("\n4. モデル訓練")
    print("-" * 30)
    
    # アンサンブルモデル（最適化済みパラメータ使用）
    ensemble = EnsembleModel()
    ensemble.train_with_optimized_params(
        X_train, y_train, 
        rf_params=optimizer.best_params.get('random_forest', {}),
        svm_params=optimizer.best_params.get('svm', {})
    )
    
    # XGBoostモデル（利用可能な場合のみ）
    xgb_model = None
    if XGBOOST_AVAILABLE:
        try:
            xgb_model = XGBoostModel(**optimizer.best_params.get('xgboost', {}))
            xgb_model.train(X_train, y_train_xgb)
        except Exception as e:
            print(f"XGBoostモデルの訓練に失敗: {e}")
            xgb_model = None
    
    # ディープラーニングモデル（利用可能な場合のみ）
    dl_model = None
    if TENSORFLOW_AVAILABLE:
        try:
            # ディープラーニング用ラベル（0始まり）
            y_train_dl = y_train - 1
            y_test_dl = y_test - 1
            dl_model = DeepLearningModel(
                input_dim=X_train.shape[1], 
                num_classes=5,
                hidden_layers=[128, 64, 32],
                dropout_rate=0.3,
                learning_rate=0.001,
                epochs=50,
                patience=10
            )
            dl_model.train(X_train, y_train_dl, X_test, y_test_dl)
        except Exception as e:
            print(f"ディープラーニングモデルの訓練に失敗: {e}")
            dl_model = None
    
    # 5. モデル評価
    print("\n5. モデル評価")
    print("-" * 30)
    
    # 各モデルの評価
    ensemble_results = ensemble.evaluate_model(X_test, y_test)
    
    # XGBoostとディープラーニングモデルの評価（利用可能な場合のみ）
    xgb_results = None
    dl_results = None
    
    if xgb_model:
        try:
            # 評価時も0始まり
            xgb_results = xgb_model.evaluate(X_test, y_test_xgb)
            # 予測値を+1して元のラベルに戻す
            xgb_results['predictions'] = xgb_results['predictions'] + 1
            xgb_results['confusion_matrix'] = confusion_matrix(y_test, xgb_results['predictions'])
        except Exception as e:
            print(f"XGBoost評価エラー: {e}")
    
    if dl_model:
        try:
            # 評価時も0始まりラベルを使用
            dl_results = dl_model.evaluate(X_test, y_test_dl)
            # 予測値を+1して元のラベルに戻す
            dl_results['predictions'] = dl_results['predictions'] + 1
            dl_results['confusion_matrix'] = confusion_matrix(y_test, dl_results['predictions'])
        except Exception as e:
            print(f"ディープラーニング評価エラー: {e}")
    
    # モデル比較
    models_dict = {
        'Ensemble': ensemble
    }
    
    if xgb_model:
        models_dict['XGBoost'] = xgb_model
    
    if dl_model:
        models_dict['Deep Learning'] = dl_model
    
    comparison_results = compare_models(models_dict, X_test, y_test)
    
    # 6. モデル保存
    print("\n6. モデル保存")
    print("-" * 30)
    
    os.makedirs('models', exist_ok=True)
    
    # アンサンブルモデル保存
    ensemble.save_model('models/optimized_ensemble_model.pickle')
    
    # XGBoostモデル保存
    if xgb_model:
        xgb_model.save_model('models/xgboost_model.pickle')
    
    # ディープラーニングモデル保存
    if dl_model:
        dl_model.save_model('models/deep_learning_model.h5')
    
    # 最適化結果保存
    optimizer.save_optimization_results('models/optimization_results.pickle')
    
    # 7. 可視化
    print("\n7. 可視化")
    print("-" * 30)
    
    # 訓練履歴の可視化
    if dl_model:
        dl_model.plot_training_history('models/training_history.pdf')
    
    # 最適化結果の可視化
    optimizer.plot_optimization_results('models/optimization_results.pdf')
    
    # 特徴量重要度の比較
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # XGBoostの特徴量重要度
    if xgb_model:
        try:
            xgb_importance = xgb_model.get_feature_importance(X_train.columns)
            xgb_importance.head(10).plot(x='feature', y='importance', kind='barh', ax=axes[0])
            axes[0].set_title('XGBoost Feature Importance')
        except Exception as e:
            print(f"XGBoost特徴量重要度の取得に失敗: {e}")
            axes[0].text(0.5, 0.5, 'XGBoost not available', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('XGBoost Feature Importance')
    else:
        axes[0].text(0.5, 0.5, 'XGBoost not available', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('XGBoost Feature Importance')
    
    # アンサンブルの特徴量重要度
    try:
        rf_importance = ensemble.get_feature_importance(X_train.columns)
        if not rf_importance.empty:
            rf_importance.head(10).plot(x='feature', y='importance', kind='barh', ax=axes[1])
            axes[1].set_title('Random Forest Feature Importance')
        else:
            axes[1].text(0.5, 0.5, 'Random Forest not available', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Random Forest Feature Importance')
    except Exception as e:
        print(f"特徴量重要度の取得に失敗: {e}")
        axes[1].text(0.5, 0.5, 'Feature importance not available', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Random Forest Feature Importance')
    
    plt.tight_layout()
    plt.savefig('models/feature_importance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()  # メモリを解放
    
    # 8. 予測テスト
    print("\n8. 予測テスト")
    print("-" * 30)
    
    # テストデータの一部で予測
    test_sample = X_test.iloc[:5]
    true_labels = y_test.iloc[:5]
    
    print("予測テスト結果:")
    print("-" * 40)
    
    for i in range(len(test_sample)):
        print(f"\nサンプル {i+1}:")
        
        # 各モデルの予測
        ensemble_pred = ensemble.predict(test_sample.iloc[i:i+1])[0]
        ensemble_prob = ensemble.predict_proba(test_sample.iloc[i:i+1])[0]
        
        xgb_pred = xgb_model.predict(test_sample.iloc[i:i+1])[0] if xgb_model else None
        xgb_prob = xgb_model.predict_proba(test_sample.iloc[i:i+1])[0] if xgb_model else None
        
        dl_pred_raw = dl_model.predict(test_sample.iloc[i:i+1])[0] if dl_model else None
        dl_pred = dl_pred_raw + 1 if dl_pred_raw is not None else None  # +1して元のラベルに戻す
        dl_prob = dl_model.predict_proba(test_sample.iloc[i:i+1])[0] if dl_model else None
        
        true_label = true_labels.iloc[i]
        
        print(f"  真のラベル: {true_label}")
        print(f"  アンサンブル: {ensemble_pred} (確率: {max(ensemble_prob):.3f})")
        if xgb_model:
            print(f"  XGBoost: {xgb_pred} (確率: {max(xgb_prob):.3f})")
        if dl_model:
            print(f"  ディープラーニング: {dl_pred} (確率: {max(dl_prob):.3f})")
    
    # モデル間の予測一致度を確認
    print(f"\nモデル間の予測一致度:")
    print("-" * 40)
    
    ensemble_preds = ensemble.predict(test_sample)
    xgb_preds = xgb_model.predict(test_sample) if xgb_model else None
    dl_preds_raw = dl_model.predict(test_sample) if dl_model else None
    dl_preds = dl_preds_raw + 1 if dl_preds_raw is not None else None  # +1して元のラベルに戻す
    
    # アンサンブル vs XGBoost
    if xgb_model:
        ensemble_xgb_agreement = np.mean(ensemble_preds == xgb_preds)
        print(f"アンサンブル vs XGBoost 一致度: {ensemble_xgb_agreement:.3f}")
    
    # アンサンブル vs ディープラーニング
    if dl_model:
        ensemble_dl_agreement = np.mean(ensemble_preds == dl_preds)
        print(f"アンサンブル vs ディープラーニング 一致度: {ensemble_dl_agreement:.3f}")
    
    # XGBoost vs ディープラーニング
    if xgb_model and dl_model:
        xgb_dl_agreement = np.mean(xgb_preds == dl_preds)
        print(f"XGBoost vs ディープラーニング 一致度: {xgb_dl_agreement:.3f}")
    
    # 全モデル一致
    if xgb_model and dl_model:
        all_agreement = np.mean((ensemble_preds == xgb_preds) & (xgb_preds == dl_preds))
        print(f"全モデル一致度: {all_agreement:.3f}")
    
    # 各モデルの正解率
    print(f"\n各モデルの正解率（テストサンプル5個）:")
    print("-" * 40)
    
    ensemble_accuracy = np.mean(ensemble_preds == true_labels)
    print(f"アンサンブル正解率: {ensemble_accuracy:.3f}")
    
    if xgb_model:
        xgb_accuracy = np.mean(xgb_preds == true_labels)
        print(f"XGBoost正解率: {xgb_accuracy:.3f}")
    
    if dl_model:
        dl_accuracy = np.mean(dl_preds == true_labels)
        print(f"ディープラーニング正解率: {dl_accuracy:.3f}")
    
    # 9. 結果サマリー
    print("\n9. 処理結果のサマリー")
    print("-" * 30)
    
    print(f"データサンプル数: {len(df)}")
    print(f"特徴量数: {len(X_train.columns)}")
    print(f"訓練データ数: {len(X_train)}")
    print(f"テストデータ数: {len(X_test)}")
    print(f"会話数: {len(train_conv_ids) + len(test_conv_ids)}")
    
    print(f"\nモデル精度:")
    for name, result in comparison_results.items():
        print(f"  {name}: {result['accuracy']:.3f}")
    
    print(f"\n保存されたモデル:")
    print(f"  - アンサンブルモデル: models/optimized_ensemble_model.pickle")
    if xgb_model:
        print(f"  - XGBoostモデル: models/xgboost_model.pickle")
    if dl_model:
        print(f"  - ディープラーニングモデル: models/deep_learning_model.h5")
    print(f"  - 最適化結果: models/optimization_results.pickle")
    print(f"\n保存されたグラフ:")
    print(f"  - 特徴量重要度比較: models/feature_importance_comparison.pdf")
    print(f"  - 最適化結果: models/optimization_results.pdf")
    if dl_model:
        print(f"  - 訓練履歴: models/training_history.pdf")
    
    print(f"\n終了時刻: {datetime.datetime.now()}")
    print("=" * 60)
    print("処理が完了しました！")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("会話活性度予測システム - 高度版")
        print("使用例:")
        print("  python main_advanced.py                    # サンプルデータで学習")
        print("  python main_advanced.py --data ./data/your_data.csv  # 指定ファイルで学習")
        print("  python main_advanced.py --test             # 予測テスト実行")
        print("  python main_advanced.py -h                 # ヘルプ表示")
        print()
    
    main() 