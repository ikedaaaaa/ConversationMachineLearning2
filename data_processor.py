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
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


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
        特徴量選択（拡張版）
        :param method: 選択方法 
            - 'percentile': フィルター法（F検定）
            - 'k_best': フィルター法（上位k個）
            - 'mutual_info': フィルター法（相互情報量）
            - 'rfe': ラッパー法（再帰的特徴量削減）
            - 'lasso': 埋め込み法（Lasso回帰）
            - 'tree_importance': 埋め込み法（木ベース重要度）
            - 'boruta': ボルタアルゴリズム（シャドウ特徴量による検定）
            - 'stepwise_forward': ステップワイズ法（前方選択）
            - 'stepwise_backward': ステップワイズ法（後方選択）
        :param kwargs: 追加パラメータ
        """
        if method == 'percentile':
            # フィルター法：F検定
            percentile = kwargs.get('percentile', 40)
            self.feature_selector = SelectPercentile(score_func=f_classif, percentile=percentile)
            print(f"フィルター法（F検定）: 上位{percentile}%の特徴量を選択")
            
        elif method == 'k_best':
            # フィルター法：上位k個
            k = kwargs.get('k', 10)
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            print(f"フィルター法（上位k個）: 上位{k}個の特徴量を選択")
            
        elif method == 'mutual_info':
            # フィルター法：相互情報量
            k = kwargs.get('k', 10)
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
            print(f"フィルター法（相互情報量）: 上位{k}個の特徴量を選択")
            
        elif method == 'rfe':
            # ラッパー法：再帰的特徴量削減
            n_features = kwargs.get('n_features', 10)
            estimator = kwargs.get('estimator', RandomForestClassifier(n_estimators=50, random_state=42))
            self.feature_selector = RFE(estimator=estimator, n_features_to_select=n_features)
            print(f"ラッパー法（RFE）: {n_features}個の特徴量を選択")
            
        elif method == 'lasso':
            # 埋め込み法：Lasso回帰
            C = kwargs.get('C', 1.0)
            estimator = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=42)
            self.feature_selector = SelectFromModel(estimator, prefit=False)
            print(f"埋め込み法（Lasso）: C={C}で特徴量を選択")
            
        elif method == 'tree_importance':
            # 埋め込み法：木ベース重要度
            threshold = kwargs.get('threshold', 'mean')
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            self.feature_selector = SelectFromModel(estimator, threshold=threshold, prefit=False)
            print(f"埋め込み法（木ベース重要度）: 閾値={threshold}で特徴量を選択")
            
        elif method == 'boruta':
            # ボルタアルゴリズム
            try:
                from boruta import BorutaPy
                n_estimators = kwargs.get('n_estimators', 'auto')
                max_iter = kwargs.get('max_iter', 100)
                
                # Random ForestをベースとしたBoruta
                rf = RandomForestClassifier(
                    n_estimators=50,  # 100から50に削減
                    n_jobs=-1, 
                    class_weight='balanced', 
                    max_depth=3,  # 5から3に削減
                    random_state=42
                )
                
                # Borutaを実行
                boruta_selector = BorutaPy(
                    rf, 
                    n_estimators=n_estimators, 
                    verbose=0, 
                    random_state=42,
                    max_iter=max_iter
                )
                
                # データをnumpy配列に変換して実行
                X_array = self.x_processed.values
                y_array = self.y_processed.values
                
                boruta_selector.fit(X_array, y_array)
                
                # 選択された特徴量を取得
                selected_features = boruta_selector.support_
                
                # カスタムセレクターを作成
                class BorutaSelector:
                    def __init__(self, support):
                        self.support_ = support
                        self.scores_ = np.ones(len(support))  # ダミースコア
                    
                    def fit_transform(self, X, y=None):
                        return X[:, self.support_]
                    
                    def transform(self, X):
                        return X[:, self.support_]
                    
                    def get_support(self):
                        return self.support_
                
                self.feature_selector = BorutaSelector(selected_features)
                print(f"ボルタアルゴリズム: シャドウ特徴量による統計的検定")
                print(f"選択された特徴量数: {np.sum(selected_features)}")
                
            except ImportError:
                print("警告: borutaライブラリがインストールされていません")
                print("インストール方法: pip install boruta")
                print("代替として木ベース重要度を使用します")
                threshold = kwargs.get('threshold', 'mean')
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                self.feature_selector = SelectFromModel(estimator, threshold=threshold, prefit=False)
                print(f"代替: 埋め込み法（木ベース重要度）: 閾値={threshold}で特徴量を選択")
            except Exception as e:
                print(f"Boruta実行エラー: {e}")
                print("代替として木ベース重要度を使用します")
                threshold = kwargs.get('threshold', 'mean')
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                self.feature_selector = SelectFromModel(estimator, threshold=threshold, prefit=False)
                print(f"代替: 埋め込み法（木ベース重要度）: 閾値={threshold}で特徴量を選択")
            
        elif method == 'stepwise_forward':
            # ステップワイズ法：前方選択
            try:
                from mlxtend.feature_selection import SequentialFeatureSelector as SFS
                k_features = kwargs.get('k_features', 'best')
                cv = kwargs.get('cv', 5)
                
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                self.feature_selector = SFS(
                    estimator=estimator,
                    k_features=k_features,
                    forward=True,
                    floating=False,
                    scoring='accuracy',
                    cv=cv,
                    n_jobs=-1
                )
                print(f"ステップワイズ法（前方選択）: k_features={k_features}, cv={cv}")
            except ImportError:
                print("警告: mlxtendライブラリがインストールされていません")
                print("インストール方法: pip install mlxtend")
                print("代替としてRFEを使用します")
                n_features = kwargs.get('n_features', 10)
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                self.feature_selector = RFE(estimator=estimator, n_features_to_select=n_features)
                print(f"代替: ラッパー法（RFE）: {n_features}個の特徴量を選択")
                
        elif method == 'stepwise_backward':
            # ステップワイズ法：後方選択
            try:
                from mlxtend.feature_selection import SequentialFeatureSelector as SFS
                k_features = kwargs.get('k_features', 'best')
                cv = kwargs.get('cv', 5)
                
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                self.feature_selector = SFS(
                    estimator=estimator,
                    k_features=k_features,
                    forward=False,
                    floating=False,
                    scoring='accuracy',
                    cv=cv,
                    n_jobs=-1
                )
                print(f"ステップワイズ法（後方選択）: k_features={k_features}, cv={cv}")
            except ImportError:
                print("警告: mlxtendライブラリがインストールされていません")
                print("インストール方法: pip install mlxtend")
                print("代替としてRFEを使用します")
                n_features = kwargs.get('n_features', 10)
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                self.feature_selector = RFE(estimator=estimator, n_features_to_select=n_features)
                print(f"代替: ラッパー法（RFE）: {n_features}個の特徴量を選択")
            
        else:
            print("無効な特徴量選択方法です")
            print("利用可能な方法: 'percentile', 'k_best', 'mutual_info', 'rfe', 'lasso', 'tree_importance', 'boruta', 'stepwise_forward', 'stepwise_backward'")
            return
        
        # 特徴量選択の実行
        self.x_processed = pd.DataFrame(
            self.feature_selector.fit_transform(self.x_processed, self.y_processed),
            index=self.x_processed.index
        )
        
        # 選択された特徴量の情報を表示
        if hasattr(self.feature_selector, 'get_support'):
            selected_features = self.feature_selector.get_support()
            print(f"特徴量選択完了: {self.x_processed.shape[1]} 特徴量（元の特徴量数: {len(selected_features)}）")
            
            # 選択された特徴量の詳細情報
            if hasattr(self.feature_selector, 'scores_'):
                feature_scores = pd.DataFrame({
                    'feature': self.x_columns,
                    'score': self.feature_selector.scores_,
                    'selected': selected_features
                }).sort_values('score', ascending=False)
                
                print("\n特徴量重要度（上位10個）:")
                print(feature_scores.head(10)[['feature', 'score', 'selected']])

    def scale_features_for_selection(self, scaler_type='standard'):
        """
        特徴量選択前のデータスケーリング
        :param scaler_type: スケーリング方法 ('standard', 'minmax', 'robust')
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
            print("StandardScalerでスケーリング")
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            print("MinMaxScalerでスケーリング")
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
            print("RobustScalerでスケーリング")
        else:
            print("無効なスケーラーです。StandardScalerを使用します")
            self.scaler = StandardScaler()
        
        # スケーリングを実行
        self.x_processed = pd.DataFrame(
            self.scaler.fit_transform(self.x_processed),
            columns=self.x_processed.columns,
            index=self.x_processed.index
        )
        print("特徴量スケーリング完了")

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

    def compare_feature_selection_methods(self, methods=['percentile', 'mutual_info', 'tree_importance'], **kwargs):
        """
        複数の特徴量選択方法を比較
        :param methods: 比較する方法のリスト
        :param kwargs: 各方法のパラメータ
        """
        results = {}
        original_features = self.x_processed.shape[1]
        
        print(f"\n特徴量選択方法の比較（元の特徴量数: {original_features}）")
        print("=" * 60)
        
        for method in methods:
            print(f"\n{method} を実行中...")
            
            # 元のデータをバックアップ
            x_backup = self.x_processed.copy()
            y_backup = self.y_processed.copy()
            
            try:
                # 特徴量選択を実行
                if method == 'percentile':
                    self.select_features(method='percentile', percentile=kwargs.get('percentile', 50))
                elif method == 'k_best':
                    self.select_features(method='k_best', k=kwargs.get('k', 10))
                elif method == 'mutual_info':
                    self.select_features(method='mutual_info', k=kwargs.get('k', 10))
                elif method == 'tree_importance':
                    self.select_features(method='tree_importance', threshold=kwargs.get('threshold', 'mean'))
                elif method == 'rfe':
                    self.select_features(method='rfe', n_features=kwargs.get('n_features', 10))
                elif method == 'lasso':
                    self.select_features(method='lasso', C=kwargs.get('C', 1.0))
                elif method == 'boruta':
                    self.select_features(method='boruta', n_estimators=kwargs.get('n_estimators', 'auto'))
                elif method == 'stepwise_forward':
                    self.select_features(method='stepwise_forward', k_features=kwargs.get('k_features', 'best'))
                elif method == 'stepwise_backward':
                    self.select_features(method='stepwise_backward', k_features=kwargs.get('k_features', 'best'))
                
                # 選択された特徴量数
                selected_count = self.x_processed.shape[1]
                
                # 簡単なモデルでクロスバリデーション評価
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                cv_scores = cross_val_score(model, self.x_processed, self.y_processed, cv=3, scoring='accuracy')
                
                results[method] = {
                    'selected_features': selected_count,
                    'reduction_ratio': (original_features - selected_count) / original_features,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"  選択された特徴量数: {selected_count}")
                print(f"  削減率: {results[method]['reduction_ratio']:.2%}")
                print(f"  クロスバリデーション精度: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                
            except Exception as e:
                print(f"  エラー: {e}")
                results[method] = None
            
            # データを復元
            self.x_processed = x_backup
            self.y_processed = y_backup
        
        # 結果の比較
        print(f"\n特徴量選択方法の比較結果:")
        print("=" * 60)
        print(f"{'方法':<15} {'特徴量数':<10} {'削減率':<10} {'精度':<15}")
        print("-" * 60)
        
        for method, result in results.items():
            if result is not None:
                print(f"{method:<15} {result['selected_features']:<10} "
                      f"{result['reduction_ratio']:<10.1%} "
                      f"{result['cv_mean']:.3f}±{result['cv_std']:.3f}")
        
        # 最良の方法を選択
        best_method = None
        best_score = -1
        
        for method, result in results.items():
            if result is not None and result['cv_mean'] > best_score:
                best_score = result['cv_mean']
                best_method = method
        
        if best_method:
            print(f"\n最良の方法: {best_method} (精度: {best_score:.3f})")
            return best_method, results
        else:
            print("\n有効な方法が見つかりませんでした")
            return None, results

    def calculate_permutation_importance(self, model, X, y, n_repeats=10, random_state=42):
        """
        Permutation Importanceを計算
        :param model: 学習済みモデル
        :param X: 特徴量
        :param y: 目的変数
        :param n_repeats: 繰り返し回数
        :param random_state: 乱数シード
        :return: 重要度のDataFrame
        """
        try:
            from sklearn.inspection import permutation_importance
            
            # Permutation Importanceを計算
            result = permutation_importance(
                model, X, y, 
                n_repeats=n_repeats, 
                random_state=random_state,
                n_jobs=-1
            )
            
            # 結果をDataFrameに変換
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std,
                'importance_rank': np.argsort(-result.importances_mean) + 1
            }).sort_values('importance_mean', ascending=False)
            
            print(f"Permutation Importance計算完了（繰り返し回数: {n_repeats}）")
            print("\n重要度（上位10個）:")
            print(importance_df.head(10)[['feature', 'importance_mean', 'importance_std', 'importance_rank']])
            
            return importance_df
            
        except ImportError:
            print("警告: sklearn.inspection.permutation_importanceが利用できません")
            print("sklearn 0.22以降が必要です")
            return None


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