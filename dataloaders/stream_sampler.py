



from __future__ import annotations
from typing import Optional, Literal
from torch.utils.data import Sampler
import torch





class StreamSampler(Sampler[int]):

    def __init__(self,
                 cfg,
                 dataset,
                 rank: Optional[int] = None,
                 world_size: Optional[int] = None,
                 drop_last: bool = False,
                 base_seed: int = 777,
                 sharding: Literal["interleave", "chunk"] = "interleave",
                 start_index: int = 0,
                 ):
        
        
        """
        各引数の説明
            dataset: 任意の Pytorch Dataset
            rank: DDP における rank
            world_size: DDP における worl size
            drop_last: データストリームでの端数の取り扱い. True なら端数切り捨て.
            base_seed: seed値
            sharding: データストリームをプロセス毎に分割する際の 'interleave'（等間引き） or 'chunk'（連続チャンク割当）を決定する
        """


        # ==============================================================
        # 基本情報の取得と検証 
        # ==============================================================
        self.dataset_len = int(len(dataset))
        if self.dataset_len < 0:
            raise ValueError("dataset length must be non-negative")


        # ==============================================================
        # DDP 関連の情報などを初期化
        # ==============================================================
        self.rank = rank
        self.world_size = int(world_size)
        self.drop_last = drop_last
        self.base_seed = base_seed
        self.sharding = sharding
        self.start_index = start_index


        # ==============================================================
        # データストリームの進行状況に関する初期化
        # ==============================================================
        self.epoch = 0  # set_epoch() で更新予定
        
        # Sampler内部のRNG（将来の摂動・シャッフル用に確保）
        self._g = torch.Generator()
        self._g.manual_seed(self.base_seed + self.rank)


        if self.sharding == "interleave":

            assert self.dataset_len % self.world_size == 0
            self._len_per_rank = int(self.dataset_len / self.world_size)

        elif self.sharding == "chunk":

            assert False
        
        else:

            assert False


    def __iter__(self):

        def _iter_once():
            if self.sharding == "interleave":
                # 等間引き: 0..N-1 を world_size ステップでスキップ
                # 例: rank r は r, r+ws, r+2*ws, ...
                produced = 0
                for i in range(self.rank, self.dataset_len, self.world_size):
                    # drop_last=True の場合は事前計算した長さ分だけ供給
                    if self.drop_last and produced >= self._len_per_rank:
                        break
                    yield i
                    produced += 1
            
            else:
                assert False
        
        for idx in _iter_once():
            yield idx






    def __len__(self) -> int:

        return self._len_per_rank


    def set_epoch(self, epoch: int) -> None:

        """
        epochの開始時に呼ばれてseed値を固定する
        """

        self.epoch = int(epoch)

        C1, C2 = 10_000_019, 1_000_000_007  # 十分大きく互いに素な定数
        new_seed = (self.base_seed + self.rank * C1 + self.epoch * C2) % (2**63 - 1)
        self._g.manual_seed(new_seed)











