
import os
import math
import random
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
        assert (os.path.exists(filelist)), '{} does not exist'.format(filelist)

        all_files = []
        task_size = []
        for i in range(self.num_task):

            # 特定タスク用の filelist からパスを読み込む
            with open(f"{filelist}/task_{i:03d}_{self.mode}.txt", 'r') as f:

                # 学習用データまでの全てのパスを取得
                task_files = f.read().splitlines()
                # print("task_files[0]: ", task_files[0])    # task_files[0]:  /home/kouyou/datasets/ImageNet21K/winter21_whole/n02689274/n02689274_12997.JPEG 0
                
                # データ分布の切り替わりを明確にするために， world_size と batch_size をもとにデータ数を削減
                file_nums = math.floor(len(task_files) / (int(cfg.ddp.world_size) * cfg.optimizer.train.batch_size)) * (int(cfg.ddp.world_size) * cfg.optimizer.train.batch_size)
                
                task_files = task_files[:int(file_nums)]

                # 後々に忘却率を測定するにはタスクの切り替わりを理解する必要があるので，各タスクのサイズをここで保存
                # （あらかじめ， world_size でプロセス毎のタスクサイズに直しておく．）
                task_size.append(len(task_files) // int(cfg.ddp.world_size))

            # タスクごとに一つの辞書に格納
            all_lines_dict[i] = task_files
        
        self.all_lines_dict = all_lines_dict
        self.task_size = task_size

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
            self.all_files = torch.stack([encode_filename(fn) for fn in all_files[:1000]])
            self.all_lables = torch.tensor(all_labels[:1000])
            assert self.all_files.shape[0] == self.all_lables.shape[0]


        # Sampler で各プロセス毎に均等にデータストリームを分割するため self.all_files と self.all_labels の長さを調整
        # length は cfg.ddp.world_size * cfg.optimizer.batch_size で割り切れる必要がある
        length = math.floor(self.all_files.shape[0] / (int(cfg.ddp.world_size) * cfg.optimizer.train.batch_size)) * (int(cfg.ddp.world_size) * cfg.optimizer.train.batch_size)
        self.all_files = self.all_files[:length]
        self.all_labels = self.all_lables[:length]


        # ==============================================================
        # タスク毎の切り替わりを示すリストを用意
        # ==============================================================
        task_period = []
        sum = 0
        for size in self.task_size:
            
            sum += size
            task_period.append(sum)

        # self.task_period[i] は i番目 のタスクの終了地点を示す．
        # batch_sampler の　db_head と対応している．
        self.task_period = task_period




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
        meta["filenames"] = fname
        meta["index"] = index
        meta["label"] = self.all_lables[index]

        out = {
            "input": image,
            "meta": meta,
        }

        return out




    def __len__(self):


        return self.all_files.shape[0]










# ==============================================================
# ImageNet21K の線形分類用データセットの定義
# ==============================================================

# 各クラスごとにシャッフル → 比率分割（決定的にするため seed 指定）
def split_by_class(lines, train_ratio: float, seed: int):
    """
    lines: ["<path> <label>", ...]
    train_ratio: 0.0 ~ 1.0
    seed: 乱数シード（決定的分割）
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)

    # パス内のスペースに耐性を持たせるため rsplit を使用
    for ln in lines:
        path, label = ln.rsplit(" ", 1)
        # print("path: ", path)
        # print("label: ", label)
        # assert False
        by_label[int(label)].append(ln)
    # print("path: ", path)
    train_lines, val_lines = [], []

    for _, items in by_label.items():
        rng.shuffle(items)
        n = len(items)

        # 四捨五入ベースで計算しつつ、片側が空にならないように調整
        n_train = int(round(n * train_ratio))
        if n >= 2:
            n_train = max(1, min(n - 1, n_train))
        else:
            n_train = 1  # n==1 の場合は train 側に入れる

        train_lines.extend(items[:n_train])
        val_lines.extend(items[n_train:])

    return train_lines, val_lines



class ImageNet21K_linear(data.Dataset):

    def __init__(self, cfg, transforms=None, filelist=None, num_task=None, train=True, linear_train=True, task_id=[], train_ratio=0.7):

        data.Dataset.__init__(self)

        
        self.train = train
        self.linear_train = linear_train
        self.filelist = filelist
        self.num_task = num_task
        self.mode = "train" if train else "val"

        self.train_ratio = train_ratio

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
        assert (os.path.exists(filelist)), '{} does not exist'.format(filelist)

        all_files = []
        task_size = []
        for i in task_id:

            # 特定タスク用の filelist からパスを読み込む
            with open(f"{filelist}/task_{i:03d}_{self.mode}.txt", 'r') as f:

                # 学習用データまでの全てのパスを取得
                task_files = f.read().splitlines()
                # print("task_files[0]: ", task_files[0])    # task_files[0]:  /home/kouyou/datasets/ImageNet21K/winter21_whole/n02689274/n02689274_12997.JPEG 0

            # タスクごとに一つの辞書に格納
            all_lines_dict[i] = task_files
        
        self.all_lines_dict = all_lines_dict
        self.task_size = task_size


        # ==============================================================
        # 全ての filelist.txt に記述されたパスを連結
        # （今後，この部分を改良してSeq-bbのようなシナリオにも対応可能にする）
        # ==============================================================
        for i in task_id:

            # 各タスクのfilelist.txtに含まれる記述を取り出す
            task_lines = self.all_lines_dict[i]

            lin_train, lin_val = split_by_class(
                task_lines,
                train_ratio=self.train_ratio,
                seed=cfg.seed + i  # タスクごとに少しずらす
            )

            if i == 0:
                print("lin_train[0:5]: ", lin_train[0:5])
                print("lin_val[0:5]: ", lin_val[0:5])

            chosen = lin_train if self.linear_train else lin_val

            # ファイルパスとラベルを分割して保存
            task_files  = [ln.rsplit(" ", 1)[0] for ln in chosen]
            task_labels = [int(ln.rsplit(" ", 1)[1]) for ln in chosen]

            all_files  += task_files
            all_labels += task_labels
        

        self.all_files = torch.stack([encode_filename(fn) for fn in all_files])
        self.all_labels = torch.tensor(all_labels)
        assert self.all_files.shape[0] == self.all_labels.shape[0]


        # ==============================================================
        # データ拡張の定義
        # ==============================================================
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms


    
    def __getitem__(self, idx):

        assert False

    
    def __len__(self):
        return self.all_files.shape[0]





