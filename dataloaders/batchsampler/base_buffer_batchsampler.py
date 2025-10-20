

from __future__ import annotations
from typing import Any, Dict, List, Optional
from torch.utils.data import Sampler

import random
from abc import ABC, abstractmethod


import torch
from torch.nn import functional as F




def gather(tensor, distributed=False):
    if not distributed:
        return [tensor]
    else:
        world_size = torch.distributed.get_world_size()
        size = tuple(tensor.shape)
        size_all = [size for _ in range(world_size)]
        torch.distributed.all_gather_object(size_all, size)

        tensor = tensor.cuda()
        max_sz = max([sz[0] for sz in size_all])
        expand_sz = tuple([max_sz] + list(size)[1:])
        tensor_all = [
            torch.zeros(size=expand_sz, dtype=tensor.dtype).cuda()
            for _ in range(world_size)
        ]
        if tensor.shape[0] < max_sz:
            pad = [0] * (2 * len(size))
            pad[-1] = max_sz - tensor.shape[0]
            tensor = F.pad(tensor, pad=pad)
        torch.distributed.all_gather(tensor_all, tensor)
        return [
            tensor_all[r][:size_all[r][0]].cpu() for r in range(world_size)
        ]


def tensorize_buffer(buffer):
    buffer_tensor = {}
    for k in buffer[0]:
        tens_list = [s[k] for s in buffer]
        if all(t is None for t in tens_list):
            continue
        dummy = [t for t in tens_list if t is not None][0] * 0.
        tens_list = [t if t is not None else dummy for t in tens_list]
        try:
            if isinstance(tens_list[0], torch.Tensor):
                tens = torch.stack(tens_list)
            elif isinstance(tens_list[0], (int, bool, float)):
                tens = torch.tensor(tens_list)
            else:
                tens = torch.tensor(tens_list)
            buffer_tensor[k] = tens
        except Exception as e:
            print(tens_list)
            print(e)
    return buffer_tensor


def gather_buffer(buffer, distributed=False):
    buffer_tensor = tensorize_buffer(buffer)
    for k in buffer_tensor:
        buffer_tensor[k] = gather(buffer_tensor[k], distributed)
    return buffer_tensor





class BaseBufferBatchSampler(Sampler[List[int]], ABC):

    """
    バッファ付き BatchSampler の抽象基底
        ー共有機能：__init__ / エントリ作成 / add_to_buffer / 基本的な状態管理
        ーフック： _evict_it_needed(), __iter__()は子クラスで定義
    """

    def __init__(self,
                 buffer_size: int,
                 repeat: int,
                 dataset,
                 sampler: Sampler[int],
                 batch_size: int,
                 rank: int = None) -> None:
        
        # 基本的なエラー確認
        assert buffer_size > 0
        assert repeat > 0
        assert batch_size > 0
        
        self.dataset = dataset
        self.sampler = sampler                 # 到着順 idx を 1 つずつ吐く（StreamSampler 等）
        self.batch_size = int(batch_size)      # この rank のバッチサイズ
        self.repeat = int(repeat)              # 何回ミニバッチを作成するか

        self.buffer_size = int(buffer_size)    # バッファの最大保持数（エントリ数）
        self.buffer_size = int(round(self.buffer_size)//self.sampler.world_size)

        self.all_indices: List[int] = list(self.sampler)
        print("self.all_indices[:10]: ", self.all_indices[:10])   # self.all_indices[:10]:  [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]


        # DDP関連
        self.distributed = torch.distributed.is_available(
        ) and torch.distributed.is_initialized()
        if rank is None:
            rank = torch.distributed.get_rank() if self.distributed else 0
        self.rank = rank


        # 学習の進行状況を確認するための変数とリスト
        self.num_batches_seen: int = 0            # iter 内で進捗を数える用（学習ループから参照）
        self.task_period = dataset.task_period

        # バッファの初期化
        self.buffer: List[Dict[str, Any]] = []
        self.db_head: int = 0                     # all_indices 上の「次に到着する」位置
        self.num_batches_yielded: int = 0         # 実際に yield したバッチ数
        self.batch_history: List[List[int]] = []  # 直近バッチの idx 記録（再開・デバッグ用）
        self.init_from_ckpt: bool = False         # ckpt 復元直後かどうかのフラグ
        self.init_from_ckpt = False               # 学習途中の記録がある場合はTrue

        # 統計の平滑化用（Polyak 等で使う想定．参考実装に合わせて用意）
        self.gamma: float = 0.5






    # ===============================================
    # 学習途中の記録，再開時の読み込み
    # ===============================================
    def state_dict(self):

        batch_history = gather(torch.tensor(self.batch_history),
                               self.distributed)
        buffer = gather_buffer(self.buffer, self.distributed)
        return {
            'buffer': buffer,
            'db_head': self.db_head,
            'num_batches_seen': self.num_batches_seen,
            'num_batches_yielded': self.num_batches_yielded,
            'batch_history': batch_history
        }

    def load_state_dict(self, state_dict):

        assert False


    # バッファにデータを格納するために，保存する情報の体裁を整えるための関数
    def _init_entry(self, idx: int) -> Dict[str, Any]:

        entry: Dict[str, Any] = {
            "idx": int(idx),
            "loss": None,                 # 後でサンプル別 loss を記録
            "feature": None,              # 後で埋め込みベクトル等を記録
            "label": None,                # 必要なら評価・分析に使用（SSLでも保持するが，学習には使用しない）
            "taskid": None,               # 必要なら評価・分析に使用
            "num_seen": 0,                # 何回バッファから取り出して学習したか
            "seen": False,                # 一度でも学習に使ったかのフラグ
            "neighbor_similarity": None,  # 
            "lifespan": 0,                # バッファに保存されていた期間
        }

        return entry

    # バッファ内にそのデータの idx が存在するかの確認
    def in_buffer(self, idx: int) -> bool:
        return int(idx) in self.buffer
    
    
    # =======================================================
    # 現在のデータストリームのタスクidを確認する関数
    # =======================================================
    def return_taskid(self):
        for i, period in enumerate(self.task_period):
            if self.db_head <= period:
                return i
        return i

    
    # =======================================================
    # これまでに確認したミニバッチの数をカウントして返す
    # =======================================================
    def advance_batches_seen(self):

        self.num_batches_seen += 1
        return self.num_batches_seen


    # =======================================================
    # データストリームから到着したデータをバッファに格納する機能
    # =======================================================
    def add_to_buffer(self, n_new:int) -> List[int]:

        """
        データストリームから到着した n_new 毎の画像をバッファに格納するだけ
        バッファ内のデータの削除は後で実行
        """

        if n_new < 0:
            return True, []
        
        added: List[int] = []

        for _ in range(n_new):

            if self.db_head >= len(self.all_indices):
                break
                
            idx = int(self.all_indices[self.db_head])
            self.db_head += 1

            if self.in_buffer(idx):
                continue
                
            entry = self._init_entry(idx)
            self.buffer.append(entry)
            added.append(idx)
        
        # lifespanを加算
        for b in self.buffer:
            b['lifespan'] += 1

        # 現在のデータストリームの タスクid を確認
        self.task_id = self.return_taskid()
        
        
        return False, added


    # ===============================================
    # ミニバッチの作成
    # ===============================================
    def sample_k(self, buffer, batch_size):
        
        if batch_size < len(buffer):
            return random.sample(buffer, k=batch_size)
        else:
            return random.choices(buffer, k=batch_size)


    def make_batch(self):

        # バッファからバッチサイズ分だけデータを取り出す
        batch = self.sample_k(self.buffer, self.batch_size)

        # print("batch: ", batch)
        # print("len(self.buffer): ", len(self.buffer))
        # print("len(batch): ", len(batch))

        # データセットの idx を取り出す
        batch_idx = [b['idx'] for b in batch]

        return batch_idx


    
    def set_epoch(self, epoch):

        self.epoch = epoch
        self.sampler.set_epoch(epoch=epoch)



    # ---------- 抽象フック ----------
    @abstractmethod
    def _resize_buffer(self) -> None:
        """buffer_size を超えている場合，何かしらの指標をもとにデータを削除する．"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self):
        pass



