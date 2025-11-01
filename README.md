# Self-Supervised Online Continual Learning (SSOCL)
自己教師ありオンライン継続学習の学習・評価コードです．
ImageNet21Kを基に作成したデータストリームで事前学習し，クラス分類や物体検出などの下流タスクで評価します．

# プログラムの全体像
学習・評価に使用するプログラムの全体像は以下の通りです．
```
SSOCL/
├── augmentations     : データ拡張関連を実装したモジュール群
├── configs           : 学習・評価の設定を記述する.yamlファイルの格納場所．
├── dataloaders       : DataLoader関連を実装したモジュール群．
├── losses            : 損失関数関連を実装したモジュール群．
├── models            : model関連を実装したモジュール群．
├── optimizers        : Optimizer関連を実装したモジュール群．
├── train             : 訓練・評価を実際に行うモジュール群．
├── main.py           : データストリームによる事前学習を実行するmainファイル．
├── main_detection.py : 物体検出による評価を実行するmainファイル．
├── main_linear.py    : 線形分類による評価を実行するmainファイル．
└── utils.py          : その他のモジュールを実装するutilsファイル．
```

# 実行方法
## ラベルなしデータストリームでの事前学習



## ラベルありデータセットでの学習と評価
### Image Classification



### Object Detection




### Segmentation


