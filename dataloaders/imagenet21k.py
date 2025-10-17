
import os
import numpy as np
from collections import defaultdict


import torch
import torchvision.datasets as datasets
import torch.utils.data as data


from utils import seed_everything





# ==============================================================
# ImageNet21K データセットの定義
# ==============================================================
class ImageNet21K(data.Dataset):

    def __init__(self, cfg, transform=None, filelist=None, num_task=None, train=True):

        data.Dataset.__init__(self)

        
        self.train = train
        self.filelist = filelist
        self.num_task = num_task
        self.mode = "train" if train else "val"

        all_files_dict = defaultdict(list)
        all_files = []

        # ==============================================================
        # seed値の固定
        # ==============================================================
        seed_everything(seed=cfg.seed)


        # ==============================================================
        # filelist.txt までのパスを設定
        # ==============================================================
        # 指定された filelist が存在するかの確認
        assert (os.path.exists(filelist)
                ), '{} does not exist'.format(filelist)

        all_files = []
        for i in range(self.num_task):

            # 特定タスク用の filelist からパスを読み込む
            with open(f"{filelist}/task_{i:03d}_{self.mode}.txt", 'r') as f:

                # 学習用データまでの全てのパスを取得
                task_files = f.read().splitlines()
                # print("task_files[0]: ", task_files[0])    # task_files[0]:  /home/kouyou/datasets/ImageNet21K/winter21_whole/n02689274/n02689274_12997.JPEG 0

            # タスクごとに一つの辞書に格納
            all_files_dict[i] = task_files
        
        self.all_files_dict = all_files_dict

        # ==============================================================
        # 全ての filelist.txt に記述されたパスを連結してデータストリームを作成
        # （今後，この部分を改良してSeq-bbのようなシナリオにも対応可能にする）
        # ==============================================================
        for i in range(self.num_task):

            task_files = self.all_files_dict[i]
            all_files += task_files
        
        self.all_files = all_files


        # ==============================================================
        # データ拡張の定義
        # ==============================================================
        if not isinstance(transform, list):
            transform = [transform]
        self.transform = transform


    def __getitem__(self, index):

        assert False




















