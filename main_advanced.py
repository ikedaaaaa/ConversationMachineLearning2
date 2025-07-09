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
    parser.add_argument('--feature-selection', '-fs', type=str, 
                       choices=['percentile', 'k_best', 'mutual_info', 'rfe', 'lasso', 'tree_importance', 'boruta', 'stepwise_forward', 'stepwise_backward'],
                       help='特徴量選択方法を指定 (指定しない場合は自動選択)')
    parser.add_argument('--use-all-data', '-all', action='store_true',
                       help='全データを使って学習（訓練・テスト分割を行わない）')
    
    args = parser.parse_args()
    
    # テストモードの処理
    if args.test:
        print("テストモード: 保存されたモデルを使用して予測テストを実行します")
        # ここにテストモードの処理を追加
        # 現在は実装されていないため、メッセージを表示して終了
        print("テストモードは現在実装されていません")
        return
    
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
    else:
        # 1. サンプルデータ作成
        print("\n1. サンプルデータ作成")
        print("-" * 30)
        df = create_sample_data_with_conversations()
        data_file = 'data/sample_data_with_conversations.csv'
    
    # 2. データ処理
    print("\n2. データ処理")
    print("-" * 30)
    data_processor = DataProcessor(data_file)
    
    # 時系列特徴量を追加（現在は無効化）
    data_processor.add_temporal_features()
    
    # 会話ごとの前処理
    data_processor.process_by_conversation()
    
    # 目的変数を設定
    data_processor.get_target_classes()
    
    # 会話単位での標準化
    data_processor.standardize_by_conversation()
    
    # 特徴量選択前のスケーリング
    print("\n特徴量選択前のスケーリング:")
    data_processor.scale_features_for_selection(scaler_type='standard')
    
    # 特徴量選択（コマンドライン引数で指定された場合は直接使用、そうでなければ比較して最適なものを選択）
    if args.feature_selection:
        print(f"\n指定された特徴量選択方法 '{args.feature_selection}' を使用:")
        if args.feature_selection == 'percentile':
            data_processor.select_features(method='percentile', percentile=50)
        elif args.feature_selection == 'k_best':
            data_processor.select_features(method='k_best', k=20)
        elif args.feature_selection == 'mutual_info':
            data_processor.select_features(method='mutual_info', k=20)
        elif args.feature_selection == 'tree_importance':
            data_processor.select_features(method='tree_importance', threshold='mean')
        elif args.feature_selection == 'rfe':
            data_processor.select_features(method='rfe', n_features=15)
        elif args.feature_selection == 'lasso':
            data_processor.select_features(method='lasso', C=1.0)
        elif args.feature_selection == 'boruta':
            data_processor.select_features(method='boruta', n_estimators=50, max_iter=50)
        elif args.feature_selection == 'stepwise_forward':
            data_processor.select_features(method='stepwise_forward', k_features='best')
        elif args.feature_selection == 'stepwise_backward':
            data_processor.select_features(method='stepwise_backward', k_features='best')
    else:
        print("\n特徴量選択方法の比較と選択:")
        best_method, selection_results = data_processor.compare_feature_selection_methods(
            methods=['percentile', 'k_best', 'mutual_info', 'rfe', 'lasso', 'tree_importance', 'boruta', 'stepwise_forward', 'stepwise_backward'],
            percentile=50,
            k=20,
            n_features=15,
            threshold='mean',
            C=1.0,
            k_features='best'
        )
        if best_method:
            print(f"\n最適な方法 '{best_method}' で特徴量選択を実行:")
            if best_method == 'percentile':
                data_processor.select_features(method='percentile', percentile=50)
            elif best_method == 'k_best':
                data_processor.select_features(method='k_best', k=20)
            elif best_method == 'mutual_info':
                data_processor.select_features(method='mutual_info', k=20)
            elif best_method == 'tree_importance':
                data_processor.select_features(method='tree_importance', threshold='mean')
            elif best_method == 'rfe':
                data_processor.select_features(method='rfe', n_features=15)
            elif best_method == 'lasso':
                data_processor.select_features(method='lasso', C=1.0)
            elif best_method == 'boruta':
                data_processor.select_features(method='boruta', n_estimators=50, max_iter=50)
            elif best_method == 'stepwise_forward':
                data_processor.select_features(method='stepwise_forward', k_features='best')
            elif best_method == 'stepwise_backward':
                data_processor.select_features(method='stepwise_backward', k_features='best')
        else:
            print("\nデフォルトの方法（相互情報量）で特徴量選択を実行:")
            data_processor.select_features(method='mutual_info', k=20)
    
    # 全データ学習モード
    if args.use_all_data:
        print("\n全データを使って学習モード")
        X_all = data_processor.df[data_processor.feature_columns]
        y_all = data_processor.df['class']
        y_all_xgb = y_all - 1

        # ハイパーパラメータ最適化
        print("\n全データでハイパーパラメータ最適化")
        optimizer = HyperparameterOptimizer()
        best_rf = optimizer.optimize_random_forest(X_all, y_all, cv=5, method='random')
        best_svm = optimizer.optimize_svm(X_all, y_all, cv=5)
        best_xgb = optimizer.optimize_xgboost(X_all, y_all_xgb, cv=5, method='grid')

        # 最適化済みパラメータでモデル学習
        ensemble = EnsembleModel()
        ensemble.train_with_optimized_params(
            X_all, y_all,
            rf_params=optimizer.best_params.get('random_forest', {}),
            svm_params=optimizer.best_params.get('svm', {})
        )
        if XGBOOST_AVAILABLE:
            try:
                xgb_model = XGBoostModel(**optimizer.best_params.get('xgboost', {}))
                xgb_model.train(X_all, y_all_xgb)
                xgb_model.save_model('models/xgboost_model_all_data.pickle')
            except Exception as e:
                print(f"XGBoostモデルの訓練に失敗: {e}")
        if TENSORFLOW_AVAILABLE:
            try:
                dl_model = DeepLearningModel(
                    input_dim=X_all.shape[1],
                    num_classes=5,
                    hidden_layers=[128, 64, 32],
                    dropout_rate=0.3,
                    learning_rate=0.001,
                    epochs=50,
                    patience=10
                )
                dl_model.train(X_all, y_all - 1, X_all, y_all - 1)
                dl_model.save_model('models/deep_learning_model_all_data.h5')
            except Exception as e:
                print(f"ディープラーニングモデルの訓練に失敗: {e}")
        # モデル保存
        ensemble.save_model('models/optimized_ensemble_model_all_data.pickle')
        optimizer.save_optimization_results('models/optimization_results_all_data.pickle')
        print("全データでハイパーパラメータ最適化＆学習したモデルを保存しました。評価や予測テストはスキップされます。")
        return
    
    # 会話単位でのデータ分割（テストサイズを調整）
    X_train, X_test, y_train, y_test, train_conv_ids, test_conv_ids = data_processor.split_by_conversation(test_size=0.3)
    
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
    # アンサンブルモデル
    ensemble_probs = ensemble.predict_proba(X_test)
    ensemble_preds_raw = np.argmax(ensemble_probs, axis=1)
    ensemble_preds = ensemble_preds_raw + 1
    
    ensemble_results = {
        'predictions': ensemble_preds,
        'probabilities': ensemble_probs,
        'accuracy': np.mean(ensemble_preds == y_test),
        'confusion_matrix': confusion_matrix(y_test, ensemble_preds)
    }
    
    # XGBoostとディープラーニングモデルの評価（利用可能な場合のみ）
    xgb_results = None
    dl_results = None
    
    if xgb_model:
        try:
            # 確率から予測を正しく取得
            xgb_probs = xgb_model.predict_proba(X_test)
            xgb_preds_raw = np.argmax(xgb_probs, axis=1)
            xgb_preds = xgb_preds_raw + 1  # +1して元のラベルに戻す
            
            # 評価結果を構築
            xgb_results = {
                'predictions': xgb_preds,
                'probabilities': xgb_probs,
                'accuracy': np.mean(xgb_preds == y_test),
                'confusion_matrix': confusion_matrix(y_test, xgb_preds)
            }
        except Exception as e:
            print(f"XGBoost評価エラー: {e}")
    
    if dl_model:
        try:
            # 確率から予測を正しく取得
            dl_probs = dl_model.predict_proba(X_test)
            dl_preds_raw = np.argmax(dl_probs, axis=1)  # 各サンプルで最も高い確率のクラスを選択
            dl_preds = dl_preds_raw + 1  # +1して元のラベルに戻す
            
            # 評価結果を構築
            dl_results = {
                'predictions': dl_preds,
                'probabilities': dl_probs,
                'accuracy': np.mean(dl_preds == y_test),
                'confusion_matrix': confusion_matrix(y_test, dl_preds)
            }
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
    
    # クラス名の定義
    class_names = ['LL', 'LM', 'MM', 'MH', 'HH']
    
    print("予測テスト結果（詳細確率付き）:")
    print("-" * 60)
    
    for i in range(len(test_sample)):
        print(f"\nサンプル {i+1}:")
        
        # 各モデルの予測（確率から正しく取得）
        # アンサンブルモデル
        ensemble_prob = ensemble.predict_proba(test_sample.iloc[i:i+1])[0]
        ensemble_pred_raw = np.argmax(ensemble_prob)
        ensemble_pred = ensemble_pred_raw + 1  # +1して元のラベルに戻す
        
        # デバッグ情報（確率と予測の一致確認）
        print(f"  デバッグ - アンサンブル:")
        print(f"    確率最大値: {max(ensemble_prob):.3f} (インデックス: {ensemble_pred_raw})")
        print(f"    予測クラス: {ensemble_pred} ({class_names[ensemble_pred_raw]})")
        
        # XGBoostモデル
        if xgb_model:
            xgb_prob = xgb_model.predict_proba(test_sample.iloc[i:i+1])[0]
            xgb_pred_raw = np.argmax(xgb_prob)
            xgb_pred = xgb_pred_raw + 1  # +1して元のラベルに戻す
            
            print(f"  デバッグ - XGBoost:")
            print(f"    確率最大値: {max(xgb_prob):.3f} (インデックス: {xgb_pred_raw})")
            print(f"    予測クラス: {xgb_pred} ({class_names[xgb_pred_raw]})")
        else:
            xgb_prob = None
            xgb_pred = None
        
        # ディープラーニングモデル
        if dl_model:
            dl_prob = dl_model.predict_proba(test_sample.iloc[i:i+1])[0]
            dl_pred_raw = np.argmax(dl_prob)
            dl_pred = dl_pred_raw + 1  # +1して元のラベルに戻す
            
            print(f"  デバッグ - ディープラーニング:")
            print(f"    確率最大値: {max(dl_prob):.3f} (インデックス: {dl_pred_raw})")
            print(f"    予測クラス: {dl_pred} ({class_names[dl_pred_raw]})")
        else:
            dl_prob = None
            dl_pred = None
        
        true_label = true_labels.iloc[i]
        true_class_name = class_names[int(true_label) - 1] if 1 <= true_label <= 5 else f"クラス{true_label}"
        
        print(f"  真のラベル: {true_label} ({true_class_name})")
        
        # アンサンブルモデルの詳細確率
        print(f"  アンサンブル予測: {ensemble_pred} ({class_names[int(ensemble_pred) - 1]})")
        print(f"    確率分布:")
        for j, (class_name, prob) in enumerate(zip(class_names, ensemble_prob)):
            marker = "★" if j == int(ensemble_pred) - 1 else "  "
            print(f"      {marker} {class_name}: {prob:.3f}")
        
        # XGBoostモデルの詳細確率
        if xgb_model:
            print(f"  XGBoost予測: {xgb_pred} ({class_names[int(xgb_pred) - 1]})")
            print(f"    確率分布:")
            for j, (class_name, prob) in enumerate(zip(class_names, xgb_prob)):
                marker = "★" if j == int(xgb_pred) - 1 else "  "
                print(f"      {marker} {class_name}: {prob:.3f}")
        
        # ディープラーニングモデルの詳細確率
        if dl_model:
            print(f"  ディープラーニング予測: {dl_pred} ({class_names[int(dl_pred) - 1]})")
            print(f"    確率分布:")
            for j, (class_name, prob) in enumerate(zip(class_names, dl_prob)):
                marker = "★" if j == int(dl_pred) - 1 else "  "
                print(f"      {marker} {class_name}: {prob:.3f}")
        
        # 予測の正誤判定
        ensemble_correct = "✓" if ensemble_pred == true_label else "✗"
        xgb_correct = "✓" if xgb_model and xgb_pred == true_label else "✗" if xgb_model else "-"
        dl_correct = "✓" if dl_model and dl_pred == true_label else "✗" if dl_model else "-"
        
        print(f"  正誤判定: アンサンブル{ensemble_correct}, XGBoost{xgb_correct}, ディープラーニング{dl_correct}")
        
        # 確率の信頼度評価
        ensemble_confidence = max(ensemble_prob)
        xgb_confidence = max(xgb_prob) if xgb_model else None
        dl_confidence = max(dl_prob) if dl_model else None
        
        # 信頼度の文字列を構築
        confidence_str = f"アンサンブル{ensemble_confidence:.3f}"
        if xgb_confidence is not None:
            confidence_str += f", XGBoost{xgb_confidence:.3f}"
        else:
            confidence_str += ", XGBoostN/A"
        if dl_confidence is not None:
            confidence_str += f", ディープラーニング{dl_confidence:.3f}"
        else:
            confidence_str += ", ディープラーニングN/A"
        
        print(f"  予測信頼度: {confidence_str}")
        
        # 確率の分散（不確実性の指標）
        ensemble_entropy = -np.sum(ensemble_prob * np.log(ensemble_prob + 1e-10))
        xgb_entropy = -np.sum(xgb_prob * np.log(xgb_prob + 1e-10)) if xgb_model else None
        dl_entropy = -np.sum(dl_prob * np.log(dl_prob + 1e-10)) if dl_model else None
        
        # 不確実性の文字列を構築
        entropy_str = f"アンサンブル{ensemble_entropy:.3f}"
        if xgb_entropy is not None:
            entropy_str += f", XGBoost{xgb_entropy:.3f}"
        else:
            entropy_str += ", XGBoostN/A"
        if dl_entropy is not None:
            entropy_str += f", ディープラーニング{dl_entropy:.3f}"
        else:
            entropy_str += ", ディープラーニングN/A"
        
        print(f"  予測不確実性: {entropy_str}")
        
        # ソフト投票（アンサンブル＋ディープラーニング）
        if dl_model:
            avg_prob = (ensemble_prob + dl_prob) / 2
            final_pred = np.argmax(avg_prob) + 1  # +1でラベルを合わせる
            print(f"  アンサンブル＋ディープラーニング投票予測: {final_pred} (確率: {avg_prob[np.argmax(avg_prob)]:.3f})")
        
        print("-" * 40)
    
    # 確率分布の可視化
    print(f"\n確率分布の可視化:")
    print("-" * 40)
    
    # サンプルごとの確率分布をプロット
    fig, axes = plt.subplots(len(test_sample), 1, figsize=(12, 3 * len(test_sample)))
    if len(test_sample) == 1:
        axes = [axes]
    
    for i in range(len(test_sample)):
        ax = axes[i]
        
        # 各モデルの確率を取得
        ensemble_prob = ensemble.predict_proba(test_sample.iloc[i:i+1])[0]
        xgb_prob = xgb_model.predict_proba(test_sample.iloc[i:i+1])[0] if xgb_model else None
        dl_prob = dl_model.predict_proba(test_sample.iloc[i:i+1])[0] if dl_model else None
        
        # プロットデータの準備
        x = np.arange(len(class_names))
        width = 0.2
        
        # 各モデルの確率をプロット
        ax.bar(x - width, ensemble_prob, width, label='アンサンブル', alpha=0.8)
        if xgb_model:
            ax.bar(x, xgb_prob, width, label='XGBoost', alpha=0.8)
        if dl_model:
            ax.bar(x + width, dl_prob, width, label='ディープラーニング', alpha=0.8)
        # アンサンブル＋ディープラーニング平均の確率分布を追加
        if dl_model:
            avg_prob = (ensemble_prob + dl_prob) / 2
            ax.bar(x + 2*width, avg_prob, width, label='アンサンブル＋DL平均', alpha=0.8, color='purple')
        
        # 真のラベルを強調表示
        true_label_idx = int(true_labels.iloc[i]) - 1
        ax.axvline(x=true_label_idx, color='red', linestyle='--', alpha=0.7, label='真のラベル')
        
        ax.set_xlabel('活性度クラス')
        ax.set_ylabel('確率')
        ax.set_title(f'サンプル {i+1} の確率分布 (真のラベル: {true_labels.iloc[i]} - {class_names[true_label_idx]})')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 確率値をテキストで表示
        for j, prob in enumerate(ensemble_prob):
            ax.text(j - width, prob + 0.01, f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
        if xgb_model:
            for j, prob in enumerate(xgb_prob):
                ax.text(j, prob + 0.01, f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
        if dl_model:
            for j, prob in enumerate(dl_prob):
                ax.text(j + width, prob + 0.01, f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
        if dl_model:
            avg_prob = (ensemble_prob + dl_prob) / 2
            for j, prob in enumerate(avg_prob):
                ax.text(j + 2*width, prob + 0.01, f'{prob:.2f}', ha='center', va='bottom', fontsize=8, color='purple')
    
    plt.tight_layout()
    plt.savefig('models/prediction_probabilities.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("確率分布の可視化を保存: models/prediction_probabilities.pdf")
    
    # 全テストデータでの予測結果をCSVに保存
    print(f"\n全テストデータでの予測結果をCSVに保存:")
    print("-" * 40)
    
    # 全テストデータでの予測と確率を取得
    all_ensemble_probs = ensemble.predict_proba(X_test)
    all_ensemble_preds_raw = np.argmax(all_ensemble_probs, axis=1)
    all_ensemble_preds = all_ensemble_preds_raw + 1
    
    all_xgb_probs = None
    all_xgb_preds = None
    if xgb_model:
        all_xgb_probs = xgb_model.predict_proba(X_test)
        all_xgb_preds_raw = np.argmax(all_xgb_probs, axis=1)
        all_xgb_preds = all_xgb_preds_raw + 1
    
    all_dl_probs = None
    all_dl_preds = None
    if dl_model:
        all_dl_probs = dl_model.predict_proba(X_test)
        all_dl_preds_raw = np.argmax(all_dl_probs, axis=1)
        all_dl_preds = all_dl_preds_raw + 1
    
    # アンサンブル＋ディープラーニング平均の確率と予測
    all_avg_probs = None
    all_avg_preds = None
    if dl_model:
        all_avg_probs = (all_ensemble_probs + all_dl_probs) / 2
        all_avg_preds_raw = np.argmax(all_avg_probs, axis=1)
        all_avg_preds = all_avg_preds_raw + 1
    
    # 結果をDataFrameにまとめる
    results_data = []
    class_names = ['LL', 'LM', 'MM', 'MH', 'HH']
    
    for i in range(len(X_test)):
        row = {
            'sample_id': i,
            'true_label': int(y_test.iloc[i]),
            'true_class': class_names[int(y_test.iloc[i]) - 1],
            
            # アンサンブルモデル
            'ensemble_prediction': int(all_ensemble_preds[i]),
            'ensemble_pred_class': class_names[int(all_ensemble_preds[i]) - 1],
            'ensemble_confidence': float(max(all_ensemble_probs[i])),
            'ensemble_entropy': float(-np.sum(all_ensemble_probs[i] * np.log(all_ensemble_probs[i] + 1e-10))),
            'ensemble_correct': all_ensemble_preds[i] == y_test.iloc[i]
        }
        
        # アンサンブルモデルの確率分布
        for j, class_name in enumerate(class_names):
            row[f'ensemble_prob_{class_name}'] = float(all_ensemble_probs[i][j])
        
        # XGBoostモデル
        if xgb_model:
            row.update({
                'xgb_prediction': int(all_xgb_preds[i]),
                'xgb_pred_class': class_names[int(all_xgb_preds[i]) - 1],
                'xgb_confidence': float(max(all_xgb_probs[i])),
                'xgb_entropy': float(-np.sum(all_xgb_probs[i] * np.log(all_xgb_probs[i] + 1e-10))),
                'xgb_correct': all_xgb_preds[i] == y_test.iloc[i]
            })
            for j, class_name in enumerate(class_names):
                row[f'xgb_prob_{class_name}'] = float(all_xgb_probs[i][j])
        else:
            row.update({
                'xgb_prediction': None,
                'xgb_pred_class': None,
                'xgb_confidence': None,
                'xgb_entropy': None,
                'xgb_correct': None
            })
            for class_name in class_names:
                row[f'xgb_prob_{class_name}'] = None
        
        # ディープラーニングモデル
        if dl_model:
            row.update({
                'dl_prediction': int(all_dl_preds[i]),
                'dl_pred_class': class_names[int(all_dl_preds[i]) - 1],
                'dl_confidence': float(max(all_dl_probs[i])),
                'dl_entropy': float(-np.sum(all_dl_probs[i] * np.log(all_dl_probs[i] + 1e-10))),
                'dl_correct': all_dl_preds[i] == y_test.iloc[i]
            })
            for j, class_name in enumerate(class_names):
                row[f'dl_prob_{class_name}'] = float(all_dl_probs[i][j])
        else:
            row.update({
                'dl_prediction': None,
                'dl_pred_class': None,
                'dl_confidence': None,
                'dl_entropy': None,
                'dl_correct': None
            })
            for class_name in class_names:
                row[f'dl_prob_{class_name}'] = None
        
        # アンサンブル＋ディープラーニング平均
        if dl_model:
            row.update({
                'ensemble_dl_avg_prediction': int(all_avg_preds[i]),
                'ensemble_dl_avg_pred_class': class_names[int(all_avg_preds[i]) - 1],
                'ensemble_dl_avg_confidence': float(max(all_avg_probs[i])),
                'ensemble_dl_avg_entropy': float(-np.sum(all_avg_probs[i] * np.log(all_avg_probs[i] + 1e-10))),
                'ensemble_dl_avg_correct': all_avg_preds[i] == y_test.iloc[i]
            })
            for j, class_name in enumerate(class_names):
                row[f'ensemble_dl_avg_prob_{class_name}'] = float(all_avg_probs[i][j])
        else:
            row.update({
                'ensemble_dl_avg_prediction': None,
                'ensemble_dl_avg_pred_class': None,
                'ensemble_dl_avg_confidence': None,
                'ensemble_dl_avg_entropy': None,
                'ensemble_dl_avg_correct': None
            })
            for class_name in class_names:
                row[f'ensemble_dl_avg_prob_{class_name}'] = None
        
        results_data.append(row)
    
    # DataFrameに変換してCSVに保存
    results_df = pd.DataFrame(results_data)
    csv_path = 'models/prediction_results_full.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"全テストデータの予測結果を保存: {csv_path}")
    print(f"保存されたサンプル数: {len(results_df)}")
    
    # 統計情報の表示
    print(f"\n全テストデータでの統計情報:")
    print("-" * 40)
    
    # 各モデルの正解率
    ensemble_accuracy = results_df['ensemble_correct'].mean()
    print(f"アンサンブル正解率: {ensemble_accuracy:.3f} ({ensemble_accuracy*100:.1f}%)")
    
    if xgb_model:
        xgb_accuracy = results_df['xgb_correct'].mean()
        print(f"XGBoost正解率: {xgb_accuracy:.3f} ({xgb_accuracy*100:.1f}%)")
    
    if dl_model:
        dl_accuracy = results_df['dl_correct'].mean()
        print(f"ディープラーニング正解率: {dl_accuracy:.3f} ({dl_accuracy*100:.1f}%)")
        
        avg_accuracy = results_df['ensemble_dl_avg_correct'].mean()
        print(f"アンサンブル＋DL平均正解率: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
    
    # 平均信頼度
    print(f"\n平均信頼度:")
    print(f"アンサンブル: {results_df['ensemble_confidence'].mean():.3f}")
    if xgb_model:
        print(f"XGBoost: {results_df['xgb_confidence'].mean():.3f}")
    if dl_model:
        print(f"ディープラーニング: {results_df['dl_confidence'].mean():.3f}")
        print(f"アンサンブル＋DL平均: {results_df['ensemble_dl_avg_confidence'].mean():.3f}")
    
    # 平均不確実性（エントロピー）
    print(f"\n平均不確実性（エントロピー）:")
    print(f"アンサンブル: {results_df['ensemble_entropy'].mean():.3f}")
    if xgb_model:
        print(f"XGBoost: {results_df['xgb_entropy'].mean():.3f}")
    if dl_model:
        print(f"ディープラーニング: {results_df['dl_entropy'].mean():.3f}")
        print(f"アンサンブル＋DL平均: {results_df['ensemble_dl_avg_entropy'].mean():.3f}")
    
    # 予測クラス分布
    print(f"\n予測クラス分布:")
    print("アンサンブル:")
    pred_dist = results_df['ensemble_pred_class'].value_counts().sort_index()
    for class_name in class_names:
        count = pred_dist.get(class_name, 0)
        ratio = count / len(results_df)
        print(f"  {class_name}: {count}個 ({ratio*100:.1f}%)")
    
    if xgb_model:
        print("XGBoost:")
        pred_dist = results_df['xgb_pred_class'].value_counts().sort_index()
        for class_name in class_names:
            count = pred_dist.get(class_name, 0)
            ratio = count / len(results_df)
            print(f"  {class_name}: {count}個 ({ratio*100:.1f}%)")
    
    if dl_model:
        print("ディープラーニング:")
        pred_dist = results_df['dl_pred_class'].value_counts().sort_index()
        for class_name in class_names:
            count = pred_dist.get(class_name, 0)
            ratio = count / len(results_df)
            print(f"  {class_name}: {count}個 ({ratio*100:.1f}%)")
        
        print("アンサンブル＋DL平均:")
        pred_dist = results_df['ensemble_dl_avg_pred_class'].value_counts().sort_index()
        for class_name in class_names:
            count = pred_dist.get(class_name, 0)
            ratio = count / len(results_df)
            print(f"  {class_name}: {count}個 ({ratio*100:.1f}%)")
    
    # 真のクラス分布
    print(f"\n真のクラス分布:")
    true_dist = results_df['true_class'].value_counts().sort_index()
    for class_name in class_names:
        count = true_dist.get(class_name, 0)
        ratio = count / len(results_df)
        print(f"  {class_name}: {count}個 ({ratio*100:.1f}%)")
    
    # モデル間の予測一致度
    print(f"\nモデル間の予測一致度:")
    if xgb_model:
        ensemble_xgb_agreement = (results_df['ensemble_prediction'] == results_df['xgb_prediction']).mean()
        print(f"アンサンブル vs XGBoost: {ensemble_xgb_agreement:.3f} ({ensemble_xgb_agreement*100:.1f}%)")
    
    if dl_model:
        ensemble_dl_agreement = (results_df['ensemble_prediction'] == results_df['dl_prediction']).mean()
        print(f"アンサンブル vs ディープラーニング: {ensemble_dl_agreement:.3f} ({ensemble_dl_agreement*100:.1f}%)")
        
        xgb_dl_agreement = (results_df['xgb_prediction'] == results_df['dl_prediction']).mean() if xgb_model else None
        if xgb_dl_agreement is not None:
            print(f"XGBoost vs ディープラーニング: {xgb_dl_agreement:.3f} ({xgb_dl_agreement*100:.1f}%)")
    
    # 高信頼度予測の分析
    print(f"\n高信頼度予測（信頼度 > 0.8）の分析:")
    high_conf_ensemble = results_df[results_df['ensemble_confidence'] > 0.8]
    if len(high_conf_ensemble) > 0:
        high_conf_accuracy = high_conf_ensemble['ensemble_correct'].mean()
        print(f"アンサンブル: {len(high_conf_ensemble)}個 ({len(high_conf_ensemble)/len(results_df)*100:.1f}%), 正解率: {high_conf_accuracy:.3f}")
    
    if xgb_model:
        high_conf_xgb = results_df[results_df['xgb_confidence'] > 0.8]
        if len(high_conf_xgb) > 0:
            high_conf_accuracy = high_conf_xgb['xgb_correct'].mean()
            print(f"XGBoost: {len(high_conf_xgb)}個 ({len(high_conf_xgb)/len(results_df)*100:.1f}%), 正解率: {high_conf_accuracy:.3f}")
    
    if dl_model:
        high_conf_dl = results_df[results_df['dl_confidence'] > 0.8]
        if len(high_conf_dl) > 0:
            high_conf_accuracy = high_conf_dl['dl_correct'].mean()
            print(f"ディープラーニング: {len(high_conf_dl)}個 ({len(high_conf_dl)/len(results_df)*100:.1f}%), 正解率: {high_conf_accuracy:.3f}")
        
        high_conf_avg = results_df[results_df['ensemble_dl_avg_confidence'] > 0.8]
        if len(high_conf_avg) > 0:
            high_conf_accuracy = high_conf_avg['ensemble_dl_avg_correct'].mean()
            print(f"アンサンブル＋DL平均: {len(high_conf_avg)}個 ({len(high_conf_avg)/len(results_df)*100:.1f}%), 正解率: {high_conf_accuracy:.3f}")
    
    # モデル間の予測一致度を確認
    print(f"\nモデル間の予測一致度:")
    if xgb_model:
        ensemble_xgb_agreement = (results_df['ensemble_prediction'] == results_df['xgb_prediction']).mean()
        print(f"アンサンブル vs XGBoost: {ensemble_xgb_agreement:.3f}")
    
    if dl_model:
        ensemble_dl_agreement = (results_df['ensemble_prediction'] == results_df['dl_prediction']).mean()
        print(f"アンサンブル vs ディープラーニング: {ensemble_dl_agreement:.3f}")
        
        xgb_dl_agreement = (results_df['xgb_prediction'] == results_df['dl_prediction']).mean() if xgb_model else None
        if xgb_dl_agreement is not None:
            print(f"XGBoost vs ディープラーニング: {xgb_dl_agreement:.3f}")
    
    # 9. 結果サマリー
    print("\n9. 処理結果のサマリー")
    print("-" * 30)
    
    print(f"データサンプル数: {len(data_processor.df)}")
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
    print(f"  - 予測確率分布: models/prediction_probabilities.pdf")
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
        print("  python main_advanced.py --feature-selection mutual_info  # 指定した特徴量選択方法で学習")
        print("  python main_advanced.py -h                 # ヘルプ表示")
        print()
    
    main() 