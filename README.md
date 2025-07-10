# ConversationMachineLearning2
会話の活性度を会話特徴量を使って機械学習させる
ちょっとミスったから削除予定

このリンクを元に作成する予定
https://github.com/ikedaaaaa/ConversationMachineLearning

## 1. コンテナを作成する．
``` bash
# yml,dockerfile,requirements.txtを編集した場合はbuildをする必要がある
docker compose -f compose.yml up --build -d

# buildをしなくてもいい場合
docker compose -f compose.yml up  -d
```

## 2. コンテナ内に入る．
``` bash
docker exec -it ConversationMachineLearning2 bash   
```

## 3. コンテナ内でpythonファイルを実行する
``` bash
# 例：python main.py 
```

## 4. コンテナを壊す
``` bash
docker compose -f compose.yml down  
```

# 開発中の注意点
## 新しいパッケージを入れたい場合
```bash
# コンテナ内で実行
pip install {package}

# 次にコンテナを立てるときにそのパッケージを入れることができるために実行
pip freeze > requirements.txt
```

# プログラムの仕様

## システム概要
会話の活性度を5段階（LL, LM, MM, MH, HH）で予測する機械学習システム

## ディレクトリ構造
```
data/
    ┝ sample_data_with_conversations.csv  # 自動生成されるサンプルデータ
    ┝ your_data.csv                      # ユーザーのデータファイル

models/
    ┝ ensemble_model_5class.pickle       # 5クラス分類モデル
    ┝ ensemble_model_LL_LM.pickle        # 2クラス分類器（LL vs LM）
    ┝ ...                                # その他の2クラス分類器
    ┝ confusion_matrix.pdf               # 混同行列
    ┝ feature_importance.pdf             # 特徴量重要度
```

## データ形式
### サンプルデータ（自動生成）
```
cid,class,speech_ratio,longest_speech_ratio,speech_speed_ratio,...
1_1,3,0.123,0.456,0.789,...
1_2,3,0.124,0.457,0.790,...
...
30_5,5,0.987,0.654,0.321,...
```

### ユーザーデータ
```
cid,class,feature1,feature2,feature3,...
00_00,3,11,0.1,0.5,...
00_01,2,13,0.5,0.3,...
...
14_59,2,21,1.0,0.8,...
```

## クラス定義
- **1 (LL)**: Low-Low（低活性度）
- **2 (LM)**: Low-Medium（低中活性度）
- **3 (MM)**: Medium-Medium（中活性度）
- **4 (MH)**: Medium-High（中高活性度）
- **5 (HH)**: High-High（高活性度）


## プログラムの処理手順

### main.py（基本システム）
1. **データ読み込み** - サンプルデータ自動作成または指定ファイル読み込み
2. **前処理**
   - 欠損値処理（平均値で補完）
   - データ標準化
   - データ離散化（20ビン）
   - 特徴量選択（上位50%）
3. **アンサンブル学習** - 5つの分類器（GaussianNB, KNN, RandomForest, GradientBoosting, SVM）
4. **2クラス分類器作成** - 全クラス組み合わせ（10個）
5. **モデル評価** - 精度、混同行列、特徴量重要度
6. **可視化** - 混同行列、特徴量重要度のPDF保存
7. **予測テスト** - テストデータでの予測結果表示

### main_advanced.py（高度システム）
1. **データ読み込み** - 同上
2. **前処理** - 同上
3. **3つのモデル学習**
   - アンサンブル学習
   - XGBoost（利用可能な場合）
   - ディープラーニング（利用可能な場合）
4. **ハイパーパラメータ最適化**
5. **モデル比較** - 3つのモデルの精度比較
6. **可視化** - 訓練履歴、最適化結果
7. **予測テスト** - 3つのモデルでの予測結果比較

## 使用方法

### 1. 基本的なシステム（main.py）

```bash
# サンプルデータで自動学習（推奨）
python main.py

# 指定ファイルで学習
python main.py --data ./data/your_data.csv

# 予測テスト実行
python main.py --test
```

### 2. 高度なシステム（main_advanced.py）

```bash
# サンプルデータで自動学習（3つのモデル比較）
python main_advanced.py

# 指定ファイルで学習
python main_advanced.py --data ./data/your_data.csv

# 予測テスト実行
python main_advanced.py --test
```

### 3. 柔軟なシステム（main_flexible.py）

```bash
# サンプルデータで実行
python main_flexible.py --create-sample

# 既存データで実行
python main_flexible.py --data ./data/your_data.csv

# カスタム設定で実行
python main_flexible.py --data ./data/your_data.csv --config config_examples/advanced_config.yaml

# カスタムサンプルデータ作成
python main_flexible.py --create-sample --sample-size 200 --sample-features 20
```

### 3. データ形式

システムは以下のデータ形式に対応しています：

#### 対応ファイル形式
- CSV (.csv)
- Excel (.xlsx, .xls)
- Parquet (.parquet)

#### カラム名の自動検出
- **会話ID**: `cid`, `conversation_id`, `conv_id`, `session_id`
- **目的変数**: `class`, `target`, `label`, `activity_level`, `impression`
- **特徴量**: 上記以外の数値カラム

#### クラス形式
- **数値クラス**: 1, 2, 3, 4, 5 など
- **文字列クラス**: "LL", "LM", "MM", "MH", "HH" など

#### 設定ファイル
YAML形式の設定ファイルで以下の項目をカスタマイズ可能：

- データ処理方法（欠損値、スケーリング、特徴量選択）
- モデル設定（アンサンブル、XGBoost、ディープラーニング）
- ハイパーパラメータ最適化
- 評価指標
- 出力設定

