#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
会話活性度予測のためのデータ処理クラス
"""

import os
import sys
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def get_class_number(class_name):
    """
    クラス番号を返す
    :param class_name: クラスの名前 (LL, LM, MM, MH, HH)
    :return: クラス番号 (1.0, 2.0, 3.0, 4.0, 5.0)
    """
    class_mapping = {
        'LL': 1.0,
        'LM': 2.0,
        'MM': 3.0,
        'MH': 4.0,
        'HH': 5.0
    }
    return class_mapping.get(class_name, None)


class DataProcessor:
    """
    会話活性度予測のためのデータ処理クラス
    """

    def __init__(self, file_path):
        """
        データ処理クラスの初期化
        :param file_path: CSVファイルのパス
        """
        # データ読み込み
        self.df = pd.read_csv(file_path)
        
        # データの基本情報
        self.columns = self.df.columns.tolist()
        
        # 欠損値処理（数値カラムのみ）
        num_cols = self.df.select_dtypes(include='number').columns
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].mean())
        
        # 説明変数と目的変数の分離
        self.x = self.df.drop(['cid', 'class'], axis=1, errors='ignore')
        self.x_columns = self.x.columns.tolist()
        self.y = self.df['class'] if 'class' in self.df.columns else None
        
        # 前処理後のデータ
        self.x_processed = None
        self.y_processed = None
        
        # 前処理のスケーラー
        self.scaler = None
        self.pca = None
        self.feature_selector = None
        
        print(f"データ読み込み完了: {len(self.df)} サンプル, {len(self.x_columns)} 特徴量")

    def get_target_classes(self, *args):
        """
        指定したクラスのデータのみを抽出
        :param args: クラス名のリスト (例: 'LL', 'MM', 'HH')
        """
        if not args:
            print("全クラスのデータを使用します")
            self.x_processed = self.x
            self.y_processed = self.y
            return
        
        # クラス番号の取得
        class_numbers = [get_class_number(name) for name in args]
        class_numbers = [num for num in class_numbers if num is not None]
        
        if not class_numbers:
            print("有効なクラス名が指定されていません")
            return
        
        # 指定したクラスのデータのみ抽出
        mask = self.y.isin(class_numbers)
        self.x_processed = self.x[mask]
        self.y_processed = self.y[mask]
        
        print(f"クラス {args} のデータを抽出: {len(self.x_processed)} サンプル")

    def handle_missing_values(self, strategy='mean'):
        """
        欠損値の処理
        :param strategy: 処理方法 ('mean', 'median', 'drop')
        """
        if strategy == 'drop':
            self.x_processed = self.x_processed.dropna()
            self.y_processed = self.y_processed[self.x_processed.index]
        elif strategy == 'median':
            self.x_processed = self.x_processed.fillna(self.x_processed.median())
        else:  # mean
            self.x_processed = self.x_processed.fillna(self.x_processed.mean())
        
        print(f"欠損値処理完了: {strategy} 方式")

    def standardize_data(self):
        """
        データの標準化
        """
        self.scaler = pre.StandardScaler()
        self.x_processed = pd.DataFrame(
            self.scaler.fit_transform(self.x_processed),
            columns=self.x_processed.columns,
            index=self.x_processed.index
        )
        print("データ標準化完了")

    def discretize_data(self, n_bins=20):
        """
        データの離散化
        :param n_bins: ビンの数
        """
        for column in self.x_processed.columns:
            self.x_processed[column] = pd.cut(
                self.x_processed[column], 
                bins=n_bins, 
                labels=False, 
                include_lowest=True
            )
        print(f"データ離散化完了: {n_bins} ビン")

    def reduce_dimensions_pca(self, n_components=0.8):
        """
       主成分分析による次元削減
        :param n_components: 主成分数または説明分散率
        """
        self.pca = PCA(n_components=n_components)
        self.x_processed = pd.DataFrame(
            self.pca.fit_transform(self.x_processed),
            index=self.x_processed.index
        )
        explained_variance_ratio = self.pca.explained_variance_ratio_.sum()
        print(f"PCA次元削減完了: {self.x_processed.shape[1]} 次元, 説明分散率: {explained_variance_ratio:.3f}")

    def select_features(self, method='percentile', **kwargs):
        """
        特徴量選択
        :param method: 選択方法 ('percentile', 'k_best', 'pca')
        :param kwargs: 追加パラメータ
        """
        if method == 'percentile':
            percentile = kwargs.get('percentile', 40)
            self.feature_selector = SelectPercentile(percentile=percentile)
        elif method == 'k_best':
            k = kwargs.get('k', 10)
            self.feature_selector = SelectKBest(k=k)
        else:
            print("無効な特徴量選択方法です")
            return
        
        self.x_processed = pd.DataFrame(
            self.feature_selector.fit_transform(self.x_processed, self.y_processed),
            index=self.x_processed.index
        )
        print(f"特徴量選択完了: {self.x_processed.shape[1]} 特徴量")

    def split_data(self, test_size=0.2, random_state=42):
        """
        データの分割
        :param test_size: テストデータの割合
        :param random_state: 乱数シード
        :return: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.x_processed, self.y_processed,
            test_size=test_size, random_state=random_state, stratify=self.y_processed
        )
        print(f"データ分割完了: 訓練データ {len(X_train)}, テストデータ {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def get_processed_data(self):
        """
        前処理済みデータの取得
        :return: (X, y)
        """
        return self.x_processed, self.y_processed

    def save_processed_data(self, file_path):
        """
        前処理済みデータの保存
        :param file_path: 保存先ファイルパス
        """
        processed_df = pd.concat([self.x_processed, self.y_processed], axis=1)
        processed_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"前処理済みデータを保存: {file_path}")

    def process_by_conversation(self, train_conversations=None):
        """
        会話ごとに前処理を行う（データリーク回避）
        :param train_conversations: 訓練用会話IDのリスト（Noneの場合は全データ）
        """
        # 会話IDを抽出
        self.df['conversation_id'] = self.df['cid'].str.split('_').str[0]
        
        if train_conversations is None:
            # 全データを使用する場合
            train_conversations = self.df['conversation_id'].unique()
        
        processed_data = []
        
        for conversation_id in train_conversations:
            # 会話ごとにデータを分割
            conv_data = self.df[self.df['conversation_id'] == conversation_id].copy()
            
            if len(conv_data) == 0:
                continue
                
            # 会話内でのみ前処理
            conv_data_processed = self._process_single_conversation(conv_data)
            processed_data.append(conv_data_processed)
        
        if processed_data:
            self.df = pd.concat(processed_data, ignore_index=True)
            print(f"会話ごとの前処理完了: {len(train_conversations)} 会話")
        else:
            print("処理対象の会話データが見つかりません")

    def _process_single_conversation(self, conv_data):
        """
        単一会話の前処理
        :param conv_data: 単一会話のデータ
        :return: 前処理済みデータ
        """
        # 数値カラムのみを対象に欠損値処理
        num_cols = conv_data.select_dtypes(include='number').columns
        conv_data[num_cols] = conv_data[num_cols].fillna(conv_data[num_cols].mean())
        
        return conv_data

    def standardize_by_conversation(self, train_conversations=None):
        """
        会話ごとに標準化を行う
        :param train_conversations: 訓練用会話IDのリスト
        """
        if train_conversations is None:
            train_conversations = self.df['conversation_id'].unique()
        
        # 訓練データの統計量を計算
        train_data = self.df[self.df['conversation_id'].isin(train_conversations)]
        self.scaler = pre.StandardScaler()
        self.scaler.fit(train_data[self.x_columns])
        
        # 全データに適用
        self.x_processed = pd.DataFrame(
            self.scaler.transform(self.x),
            columns=self.x_columns,
            index=self.x.index
        )
        
        # y_processedを確実に設定
        if hasattr(self, 'y') and self.y is not None:
            self.y_processed = self.y
        else:
            # yが設定されていない場合は、classカラムから取得
            self.y_processed = self.df['class']
        
        print("会話単位での標準化完了")

    def split_by_conversation(self, test_size=0.2, random_state=42):
        """
        会話単位でデータを分割
        :param test_size: テストデータの割合
        :param random_state: 乱数シード
        :return: (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # 会話IDのリスト
        conversation_ids = self.df['conversation_id'].unique()
        
        # 会話ごとのクラス分布を確認
        conv_class_dist = self.df.groupby('conversation_id')['class'].first()
        print(f"会話ごとのクラス分布: {conv_class_dist.value_counts().to_dict()}")
        
        # クラス数とテストサイズを確認
        num_classes = len(conv_class_dist.value_counts())
        num_test_conversations = max(1, int(len(conversation_ids) * test_size))
        
        print(f"クラス数: {num_classes}, テスト会話数: {num_test_conversations}")
        
        # stratifyが可能かチェック
        min_class_count = conv_class_dist.value_counts().min()
        
        # テストサイズがクラス数より小さい場合や、最小クラス数が2未満の場合はstratifyなし
        if num_test_conversations < num_classes or min_class_count < 2:
            print(f"警告: テストサイズ({num_test_conversations}) < クラス数({num_classes}) または 最小クラス数({min_class_count}) < 2 のため、stratifyなしで分割します")
            train_conv_ids, test_conv_ids = train_test_split(
                conversation_ids, 
                test_size=test_size, 
                random_state=random_state
            )
        else:
            # 会話単位で分割（stratifyあり）
            train_conv_ids, test_conv_ids = train_test_split(
                conversation_ids, 
                test_size=test_size, 
                random_state=random_state,
                stratify=conv_class_dist
            )
        
        # 訓練データとテストデータを抽出
        train_mask = self.df['conversation_id'].isin(train_conv_ids)
        test_mask = self.df['conversation_id'].isin(test_conv_ids)
        
        X_train = self.x_processed[train_mask]
        X_test = self.x_processed[test_mask]
        y_train = self.y_processed[train_mask]
        y_test = self.y_processed[test_mask]
        
        print(f"会話単位でのデータ分割完了: 訓練会話 {len(train_conv_ids)}, テスト会話 {len(test_conv_ids)}")
        print(f"訓練データ {len(X_train)}, テストデータ {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, train_conv_ids, test_conv_ids

    def add_temporal_features(self):
        """
        # 時系列特徴量を追加（現在は無効化）
        # self.df['conversation_id'] = self.df['cid'].str.split('_').str[0]
        # self.df['interval_id'] = self.df['cid'].str.split('_').str[1].astype(int)
        # self.df['interval_order'] = self.df.groupby('conversation_id')['interval_id'].rank()
        # self.df['conversation_progress'] = (self.df['interval_order'] - 1) / (
        #     self.df.groupby('conversation_id')['interval_id'].transform('count') - 1
        # )
        # for col in self.x_columns:
        #     if col in self.df.columns:
        #         self.df[f'{col}_diff'] = self.df.groupby('conversation_id')[col].diff()
        #         self.df[f'{col}_diff'] = self.df[f'{col}_diff'].fillna(0)
        # for col in self.x_columns:
        #     if col in self.df.columns:
        #         self.df[f'{col}_ma3'] = self.df.groupby('conversation_id')[col].rolling(
        #             window=3, min_periods=1
        #         ).mean().reset_index(0, drop=True)
        # self.df['conversation_length'] = self.df.groupby('conversation_id')['interval_id'].transform('count')
        # self.x_columns = [col for col in self.df.columns 
        #                  if col not in ['cid', 'class', 'conversation_id', 'interval_id']]
        # print(f"時系列特徴量追加完了: {len(self.x_columns)} 特徴量")
        # print(f"追加された特徴量: interval_order, conversation_progress, *_diff, *_ma3, conversation_length")
        """
        pass



def main():
    """
    テスト用のメイン関数
    """
    # データ処理のテスト
    processor = DataProcessor('./data/combined_features_with_activity.csv')
    
    # 全クラスのデータを使用
    processor.get_target_classes()
    
    # 前処理
    processor.handle_missing_values()
    processor.standardize_data()
    processor.select_features(method='percentile', percentile=50)
    
    # データ分割
    X_train, X_test, y_train, y_test = processor.split_data()
    
    print("データ処理テスト完了")


if __name__ == '__main__':
    main() 