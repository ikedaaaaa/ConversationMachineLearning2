#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
会話活性度予測メインプログラム
READMEの仕様に従って機械学習モデルを作成する
"""

import os
import sys
import datetime
import pickle
import numpy as np
import argparse
import pandas as pd
from data_processor import DataProcessor
from ensemble_model import EnsembleModel, create_binary_classifiers

SAMPLE_DATA_PATH = './data/sample_data_with_conversations.csv'

def create_sample_data_with_conversations():
    """
    会話IDを含むサンプルデータを作成（main_advanced.pyと同じ仕様）
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

def create_sample_data(file_path):
    """
    サンプルデータを作成（後方互換性のため）
    """
    return create_sample_data_with_conversations()

def main():
    parser = argparse.ArgumentParser(description='会話活性度予測システム')
    parser.add_argument('--data', '-d', type=str, 
                       default=None,
                       help='学習データファイルのパス (デフォルト: サンプルデータ)')
    parser.add_argument('--test', '-t', action='store_true',
                       help='予測テストを実行する')
    args = parser.parse_args()

    # 引数なしで実行された場合はサンプルデータで学習
    if args.data is None:
        print("引数なしで実行されました。サンプルデータで学習を開始します。")
        print("使用例:")
        print("  python main.py                    # サンプルデータで学習")
        print("  python main.py --data ./data/your_data.csv  # 指定ファイルで学習")
        print("  python main.py --test             # 予測テスト実行")
        print("  python main.py -h                 # ヘルプ表示")
        print("=" * 60)
        data_file = SAMPLE_DATA_PATH
        if not os.path.exists(data_file):
            create_sample_data(data_file)
    else:
        data_file = args.data

    print("=" * 60)
    print("会話活性度予測システム")
    print("=" * 60)
    print(f"開始時刻: {datetime.datetime.now()}")

    # テストモードの場合は予測テストを実行
    if args.test:
        test_prediction(data_file)
        return

    # 1. データを読み込む
    print("\n1. データを読み込む")
    print("-" * 30)
    if not os.path.exists(data_file):
        print(f"エラー: データファイルが見つかりません: {data_file}")
        return
    processor = DataProcessor(data_file)
    
    # 2. データの前処理を行う
    print("\n2. データの前処理を行う")
    print("-" * 30)
    
    # 2.1. 欠損値の処理
    print("2.1. 欠損値の処理")
    processor.get_target_classes()  # 全クラスのデータを使用
    processor.handle_missing_values(strategy='mean')
    
    # 2.2. データの標準化
    print("\n2.2. データの標準化")
    processor.standardize_data()
    
    # 2.3. データの離散化
    print("\n2.3. データの離散化")
    processor.discretize_data(n_bins=20)
    
    # 2.4. 次元削減を行う（特徴量選択）
    print("\n2.4. 次元削減を行う（特徴量選択）")
    processor.select_features(method='percentile', percentile=50)
    
    # データ分割
    X_train, X_test, y_train, y_test = processor.split_data(test_size=0.2, random_state=42)
    
    # 3. アンサンブル学習を行う
    print("\n3. アンサンブル学習を行う")
    print("-" * 30)
    
    ensemble = EnsembleModel()
    ensemble.create_ensemble_model(voting='soft')
    ensemble.train_model(X_train, y_train, optimize_hyperparams=False)
    
    # 4. モデルの評価
    print("\n4. モデルの評価")
    print("-" * 30)
    
    results = ensemble.evaluate_model(X_test, y_test)
    
    # クロスバリデーション
    X, y = processor.get_processed_data()
    cv_scores = ensemble.cross_validate_model(X, y)
    
    # 5. モデルを保存する
    print("\n5. モデルを保存する")
    print("-" * 30)
    
    # 保存ディレクトリの作成
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)
    
    # 5クラス分類モデルの保存
    model_path = f"{model_dir}/ensemble_model_5class.pickle"
    ensemble.save_model(model_path)
    
    # 2クラス分類器の作成と保存
    print("\n2クラス分類器の作成")
    create_binary_classifiers(processor, model_dir)
    
    # 6. 結果の可視化
    print("\n6. 結果の可視化")
    print("-" * 30)
    
    # 混同行列の可視化
    confusion_matrix_path = f"{model_dir}/confusion_matrix.pdf"
    ensemble.plot_confusion_matrix(save_path=confusion_matrix_path)
    
    # 特徴量重要度の可視化
    feature_names = processor.x_processed.columns.tolist()
    feature_importance_path = f"{model_dir}/feature_importance.pdf"
    ensemble.plot_feature_importance(feature_names, save_path=feature_importance_path)
    
    # 7. 処理結果のサマリー
    print("\n7. 処理結果のサマリー")
    print("-" * 30)
    print(f"データサンプル数: {len(processor.df)}")
    print(f"特徴量数: {len(processor.x_columns)}")
    print(f"前処理後特徴量数: {len(processor.x_processed.columns)}")
    print(f"訓練データ数: {len(X_train)}")
    print(f"テストデータ数: {len(X_test)}")
    print(f"最終精度: {results['accuracy']:.3f}")
    print(f"クロスバリデーション平均精度: {cv_scores.mean():.3f}")
    print(f"保存されたモデル数: 11個（5クラス分類1個 + 2クラス分類10個）")
    print(f"保存されたグラフ:")
    print(f"  - 混同行列: {confusion_matrix_path}")
    print(f"  - 特徴量重要度: {feature_importance_path}")
    
    # 8. 予測テスト（自動実行）
    print("\n8. 予測テスト")
    print("-" * 30)
    
    # テストデータの一部で予測
    test_sample = X_test.iloc[:5]
    true_labels = y_test.iloc[:5]
    
    print("予測テスト結果:")
    print("-" * 40)
    
    for i in range(len(test_sample)):
        pred = ensemble.predict(test_sample.iloc[i:i+1])[0]
        prob = ensemble.predict_proba(test_sample.iloc[i:i+1])[0]
        true_label = true_labels.iloc[i]
        
        # クラス名に変換
        class_names = {1: 'LL', 2: 'LM', 3: 'MM', 4: 'MH', 5: 'HH'}
        pred_name = class_names.get(pred, str(pred))
        true_name = class_names.get(true_label, str(true_label))
        
        print(f"サンプル {i+1}:")
        print(f"  真のラベル: {true_name} ({true_label})")
        print(f"  予測クラス: {pred_name} ({pred}) (確率: {max(prob):.3f})")
        print()
    
    # 正解率の計算
    predictions = ensemble.predict(test_sample)
    accuracy = np.mean(predictions == true_labels)
    print(f"予測テスト正解率（5サンプル）: {accuracy:.3f}")
    
    print(f"\n終了時刻: {datetime.datetime.now()}")
    print("=" * 60)
    print("処理が完了しました！")


def test_prediction(data_file='./data/combined_features_with_activity.csv'):
    """
    予測テスト
    """
    print("予測テスト")
    print("-" * 30)
    
    # データ処理を実行してテストデータを準備
    processor = DataProcessor(data_file)
    processor.get_target_classes()
    processor.handle_missing_values()
    processor.standardize_data()
    processor.discretize_data(n_bins=20)
    processor.select_features(method='percentile', percentile=50)
    
    X_train, X_test, y_train, y_test = processor.split_data()
    
    # 5クラス分類モデルを読み込み
    ensemble_model_path = './models/ensemble_model_5class.pickle'
    if os.path.exists(ensemble_model_path):
        with open(ensemble_model_path, 'rb') as f:
            ensemble_model = pickle.load(f)
        print(f"モデルを読み込み: {ensemble_model_path}")
        
        # テストデータで予測
        test_sample = X_test.iloc[:5]
        true_labels = y_test.iloc[:5]
        
        print("予測結果:")
        for i in range(len(test_sample)):
            pred = ensemble_model.predict(test_sample.iloc[i:i+1])[0]
            prob = ensemble_model.predict_proba(test_sample.iloc[i:i+1])[0]
            true_label = true_labels.iloc[i]
            
            # クラス名に変換
            class_names = {1: 'LL', 2: 'LM', 3: 'MM', 4: 'MH', 5: 'HH'}
            pred_name = class_names.get(pred, str(pred))
            
            print(f"サンプル {i+1}: 予測クラス = {pred_name} (確率: {max(prob):.3f})")
    else:
        print(f"モデルファイルが見つかりません: {ensemble_model_path}")
    
    # XGBoostモデルの予測テスト（存在する場合）
    xgb_model_path = './models/xgboost_model.pickle'
    if os.path.exists(xgb_model_path):
        try:
            with open(xgb_model_path, 'rb') as f:
                xgb_model = pickle.load(f)
            print(f"\nXGBoostモデルを読み込み: {xgb_model_path}")
            
            xgb_preds = xgb_model.predict(test_sample)
            xgb_probs = xgb_model.predict_proba(test_sample)
            
            print("XGBoost予測結果:")
            for i in range(len(test_sample)):
                pred = xgb_preds[i]
                prob = xgb_probs[i]
                class_names = {1: 'LL', 2: 'LM', 3: 'MM', 4: 'MH', 5: 'HH'}
                pred_name = class_names.get(pred, str(pred))
                print(f"サンプル {i+1}: 予測クラス = {pred_name} (確率: {max(prob):.3f})")
        except Exception as e:
            print(f"XGBoostモデルの読み込みエラー: {e}")
    
    # ディープラーニングモデルの予測テスト（存在する場合）
    dl_model_path = './models/deep_learning_model.h5'
    if os.path.exists(dl_model_path):
        try:
            import tensorflow as tf
            dl_model = tf.keras.models.load_model(dl_model_path)
            print(f"\nディープラーニングモデルを読み込み: {dl_model_path}")
            
            dl_probs = dl_model.predict(test_sample)
            dl_preds = np.argmax(dl_probs, axis=1) + 1  # 1-5のクラスに変換
            
            print("ディープラーニング予測結果:")
            for i in range(len(test_sample)):
                pred = dl_preds[i]
                prob = dl_probs[i]
                class_names = {1: 'LL', 2: 'LM', 3: 'MM', 4: 'MH', 5: 'HH'}
                pred_name = class_names.get(pred, str(pred))
                print(f"サンプル {i+1}: 予測クラス = {pred_name} (確率: {max(prob):.3f})")
        except Exception as e:
            print(f"ディープラーニングモデルの読み込みエラー: {e}")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("会話活性度予測システム")
        print("使用例:")
        print("  python main.py                    # サンプルデータで学習")
        print("  python main.py --data ./data/your_data.csv  # 指定ファイルで学習")
        print("  python main.py --test             # 予測テスト実行")
        print("  python main.py -h                 # ヘルプ表示")
        print()
    
    main() 