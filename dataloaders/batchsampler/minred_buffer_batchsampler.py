
import math
import random
import numpy as np
from collections import deque


import torch



from dataloaders.batchsampler.base_buffer_batchsampler import BaseBufferBatchSampler




class MinRedBufferBatchSampler(BaseBufferBatchSampler):

    """
    MinRedバッファを備えたBatchSsampler
    """


    def __init__(self,
                 buffer_size: int,
                 repeat: int,
                 dataset,
                 sampler,
                 batch_size,
                ) -> None:
        
        super().__init__(buffer_size, repeat, dataset, sampler, batch_size)


    def _evict_until_fit(self) -> None:
        """buffer_size を超えている場合，何かしらの指標をもとにデータを削除する．"""
        pass



    def __iter__(self):
        
        # 学習途中かどうかの確認．学習途中の記録がなければ各変数を初期化
        if not self.init_from_ckpt:
            self.db_head = 0
            self.num_batches_seen = 0
            self.num_batches_yielded = 0
            self.batch_history = deque(maxlen=128)
        

        # バッファサイズが全データ数よりも少ないかを確認
        assert self.buffer_size <= len(self.all_indices)

        # 
        while self.num_batches_yielded < len(self):

            # self.bufferにデータを追加
            indices_to_add = self.add_to_buffer(self.batch_size)
            self.indices_to_add = indices_to_add

            # バッファ内にデータが一定量たまるまで学習を行わずデータを溜め続ける
            if len(self.buffer) < self.buffer_size:
                continue

            # repeat 回繰り返してミニバッチを作成
            for j in range(self.repeat):

                # ミニバッチの作成
                batch_idx = self.make_batch()
                # print("batch_idx: ", batch_idx)

                self.num_batches_yielded += 1
                self.batch_history += [batch_idx]

                yield batch_idx



    def __len__(self) -> int:
        return len(self.sampler) * self.repeat // self.batch_size


