#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
データ処理クラスのテストスクリプト
"""

from data_processor import DataProcessor


def test_data_processor():
    """
    データ処理クラスのテスト
    """
    print("データ処理クラスのテスト開始")
    print("=" * 50)
    
    # データ処理クラスの初期化
    processor = DataProcessor('./data/combined_features_with_activity.csv')
    
    # データの基本情報を表示
    print(f"データサンプル数: {len(processor.df)}")
    print(f"特徴量数: {len(processor.x_columns)}")
    print(f"特徴量名: {processor.x_columns}")
    print(f"クラス分布:\n{processor.y.value_counts().sort_index()}")
    
    # 全クラスのデータを使用
    processor.get_target_classes()
    
    # 前処理のテスト
    print("\n前処理のテスト")
    print("-" * 30)
    
    # 欠損値処理
    processor.handle_missing_values()
    print("✓ 欠損値処理完了")
    
    # 標準化
    processor.standardize_data()
    print("✓ 標準化完了")
    
    # 離散化
    processor.discretize_data(n_bins=10)
    print("✓ 離散化完了")
    
    # 特徴量選択
    processor.select_features(method='percentile', percentile=50)
    print("✓ 特徴量選択完了")
    
    # データ分割
    X_train, X_test, y_train, y_test = processor.split_data()
    print(f"✓ データ分割完了: 訓練データ {len(X_train)}, テストデータ {len(X_test)}")
    
    print("\nデータ処理テスト完了！")


if __name__ == '__main__':
    test_data_processor() 