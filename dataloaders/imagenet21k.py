
import os
import math
import numpy as np
from collections import defaultdict


import torch
import torchvision.datasets as datasets
import torch.utils.data as data


from utils import seed_everything




def encode_filename(fn, max_len=200):
    assert len(
        fn
    ) < max_len, f"Filename is too long. Specified max length is {max_len}"
    fn = fn + '\n' + ' ' * (max_len - len(fn))
    fn = np.fromstring(fn, dtype=np.uint8)
    fn = torch.ByteTensor(fn)
    return fn


def decode_filename(fn):
    fn = fn.cpu().numpy().astype(np.uint8)
    fn = fn.tostring().decode('utf-8')
    fn = fn.split('\n')[0]
    return fn



# ==============================================================
# ImageNet21K データセットの定義
# ==============================================================
class ImageNet21K(data.Dataset):

    def __init__(self, cfg, transforms=None, filelist=None, num_task=None, train=True):

        data.Dataset.__init__(self)

        
        self.train = train
        self.filelist = filelist
        self.num_task = num_task
        self.mode = "train" if train else "val"

        all_lines_dict = defaultdict(list)
        all_files = []
        all_labels = []

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
            all_lines_dict[i] = task_files
        
        self.all_lines_dict = all_lines_dict

        # ==============================================================
        # 全ての filelist.txt に記述されたパスを連結
        # （今後，この部分を改良してSeq-bbのようなシナリオにも対応可能にする）
        # ==============================================================
        for i in range(self.num_task):

            # 各タスクのfilelist.txtに含まれる記述を取り出す
            task_lines = self.all_lines_dict[i]

            # ファイルパスとラベルを分割して保存
            task_files = [fn.split(" ")[0] for fn in task_lines]
            task_labels = [int(fn.split(" ")[-1]) for fn in task_lines]
            
            
            all_files += task_files
            all_labels += task_labels


        # debug時は使用するデータ数を削減する
        if not cfg.debug:
            self.all_files = torch.stack([encode_filename(fn) for fn in all_files])
            self.all_lables = torch.tensor(all_labels)
            assert self.all_files.shape[0] == self.all_lables.shape[0]
        else:
            self.all_files = torch.stack([encode_filename(fn) for fn in all_files[:2000]])
            self.all_lables = torch.tensor(all_labels[:2000])
            assert self.all_files.shape[0] == self.all_lables.shape[0]


        # Sampler で各プロセス毎に均等にデータストリームを分割するため self.all_files と self.all_labels の長さを調整
        # length は cfg.ddp.world_size * cfg.optimizer.batch_size で割り切れる必要がある
        length = math.floor(self.all_files.shape[0] / (int(cfg.ddp.world_size) * cfg.optimizer.train.batch_size)) * (int(cfg.ddp.world_size) * cfg.optimizer.train.batch_size)
        self.all_files = self.all_files[:length]
        self.all_labels = self.all_lables[:length]


        # ==============================================================
        # データ拡張の定義
        # ==============================================================
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms


    def __getitem__(self, index):

        MAX_TRIES = 50
        for i in range(MAX_TRIES):
            try:
                fname = decode_filename(self.all_files[index])
                image = datasets.folder.pil_loader(fname)
                # label = self.all_labels[index]
                break
            except Exception:
                if i == MAX_TRIES - 1:
                    raise ValueError(
                        f'Aborting. Failed to load {MAX_TRIES} times in a row. Check {fname}'
                    )
                print(f'Failed to load. {fname}')
                index = np.random.randint(len(self))
        

        meta = {}
        i = 0

        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        if transform is not None:
            image = transform(image)
        

        meta["transid"] = i
        meta["filename"] = fname
        meta["index"] = index
        meta["label"] = self.all_lables[index]

        out = {
            "input": image,
            "meta": meta,
        }

        return out




    def __len__(self):


        return self.all_files.shape[0]














