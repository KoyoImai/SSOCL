# Self-Supervised Online Continual Learning (SSOCL)
自己教師ありオンライン継続学習の学習・評価コードです．
ImageNet21Kを基に作成したデータストリームで事前学習し，クラス分類や物体検出などの下流タスクで評価します．

# プログラムの全体像
学習・評価に使用するプログラムの全体像は以下の通りです．セグメンテーションなどの評価プログラムはこれから実装する予定です．
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

ImageNet21Kデータストリームで事前学習を実行する．
学習の設定は`configs`ディレクトリの下に`.yaml`ファイルを追加，修正することで変更できます．
`.yaml`ファイルを追加して学習する場合は，実行時の`--config-path`と`--config-name`を修正してから実行してください．

- MinRed ([paper](https://arxiv.org/pdf/2203.12710)):
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --config-path ./configs/default/ --config-name default_minred
    ```

- EMP-SSL ([paper](https://arxiv.org/pdf/2304.03977)):
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --config-path ./configs/default/ --config-name default_empssl
    ```

- Imai ([paper](https://openaccess.thecvf.com/content/ACCV2024/papers/Imai_Faster_convergence_and_Uncorrelated_gradients_in_Self-Supervised_Online_Continual_Learning_ACCV_2024_paper.pdf)):
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --config-path ./configs/default/ --config-name default_ours
    ```

## ラベルありデータセットでの学習と評価
### Image Classification
クラス分類による下流タスクの性能評価を実行する．\\
評価設定を記述した`.yaml`ファイルは，事前学習時と同じファイルを使用しますが，今後事前学習と評価は別の`.yaml`ファイルにする予定です．\\
事前学習とは異なり，基本的に1gpuでの評価を想定しています．\\
（もしかしたら，Multi GPUでも評価可能になるかもしれません．）

- MinRed ([paper](https://arxiv.org/pdf/2203.12710)):
    ```
    python main_linear.py --config-path ./configs/default/ --config-name default_minred
    ```

- EMP-SSL ([paper](https://arxiv.org/pdf/2304.03977)):
    ```
    python main_linear.py --config-path ./configs/default/ --config-name default_empssl
    ```

- Imai ([paper](https://openaccess.thecvf.com/content/ACCV2024/papers/Imai_Faster_convergence_and_Uncorrelated_gradients_in_Self-Supervised_Online_Continual_Learning_ACCV_2024_paper.pdf)):
    ```
    python main_linear.py --config-path ./configs/default/ --config-name default_ours
    ```


### Object Detection




### Segmentation


