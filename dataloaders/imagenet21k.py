
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

                task_files = f.read().splitlines()
















