# ConversationMachineLearning2
会話の活性度を会話特徴量を使って機械学習させる

このリンクを元に作成
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
## ディレクトリ構造
```
data/
    ┝ conv_feature_value_data/
    │    └ 00_feature_value_data.csv
    │    └ 01_feature_value_data.csv
    │    ...
    ┝ correct_activity_label.csv
```
## ファイルの中身
```
00_feature_value_data.csv
cid,word_count,rms_avg,.....
00_00,11,0.1,.....
00_01,13,0.5,.....
...
00_59,21,1.0,.....

correct_activity_label.csv
cid,class
00_00,3.0
00_01,1.0
00_02,3.0
00_03,1.0
...
14_59,3.0
```
## したいこと
特徴量と活性度のラベルを紐付け，機械学習を行い，活性度予測モデルを作成する


## プログラムの簡単な手順
### 1.データを読み込む
