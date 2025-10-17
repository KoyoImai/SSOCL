

from __future__ import annotations
from typing import Any, Dict, List, Optional
from torch.utils.data import Sampler

from abc import ABC, abstractmethod



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
                 batch_size: int) -> None:
        
        # 基本的なエラー確認
        assert buffer_size > 0
        assert repeat > 0
        assert batch_size > 0
        
        self.dataset = dataset
        self.sampler = sampler                 # 到着順 idx を 1 つずつ吐く（StreamSampler 等）
        self.batch_size = int(batch_size)      # この rank のバッチサイズ
        self.buffer_size = int(buffer_size)    # バッファの最大保持数（エントリ数）
        self.repeat = int(repeat)              # 何回ミニバッチを作成するか

        self.all_indices: List[int] = list(self.sampler)
        # print("self.all_indices[:10]: ", self.all_indices[:10])   # self.all_indices[:10]:  [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]

        
        # バッファの初期化
        self.buffer: List[Dict[str, Any]] = []
        self.db_head: int = 0                     # all_indices 上の「次に到着する」位置
        self.num_batches_seen: int = 0            # iter 内で進捗を数える用（学習ループから参照）
        self.num_batches_yielded: int = 0         # 実際に yield したバッチ数
        self.batch_history: List[List[int]] = []  # 直近バッチの idx 記録（再開・デバッグ用）
        self.init_from_ckpt: bool = False         # ckpt 復元直後かどうかのフラグ

        # 統計の平滑化用（Polyak 等で使う想定．参考実装に合わせて用意）
        self.gamma: float = 0.5



    # バッファにデータを格納するために，保存する情報の体裁を整えるための関数
    def _init_entry(self, idx: int) -> Dict[str, Any]:

        entry: Dict[str, Any] = {
            "idx": int(idx),
            "loss": None,           # 後でサンプル別 loss を記録
            "feature": None,        # 後で埋め込みベクトル等を記録
            "label": None,          # 必要なら評価・統計用に付与（SSLでも保持するが，学習には使用しない）
            "num_seen": 0,          # 何回バッファから取り出して学習したか
            "seen": False,          # 一度でも学習に使ったかのフラグ
            "lifespan": 0,          # バッファに滞在したイテレーション数
        }

        return entry

    # バッファ内にそのデータの idx が存在するかの確認
    def in_buffer(self, idx: int) -> bool:
        return int(idx) in self.buffer



    # データストリームから到着したデータをバッファに格納する機能
    def add_to_buffer(self, n_new:int) -> List[int]:

        """
        データストリームから到着した n_new 毎の画像をバッファに格納するだけ
        バッファ内のデータの削除は後で実行
        """

        if n_new < 0:
            return []
        
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

        
        return added


    # ---------- 抽象フック ----------
    @abstractmethod
    def _evict_until_fit(self) -> None:
        """buffer_size を超えている場合，何かしらの指標をもとにデータを削除する．"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self):
        pass



