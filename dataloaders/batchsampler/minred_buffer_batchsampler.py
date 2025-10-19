
import math
import random
import numpy as np
from collections import deque


import torch
import torch.nn.functional as F



from dataloaders.batchsampler.base_buffer_batchsampler import BaseBufferBatchSampler



def tensorize_buffer(buffer):
    
    buffer_tensor = {}
    for k in buffer[0]:
        
        tens_list = [s[k] for s in buffer]
        if all(t is None for t in tens_list):
            continue
        if k == "fn":
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
            print(e)
    
    return buffer_tensor




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



    def _resize_buffer(self) -> None:
        """buffer_size を超えている場合，何かしらの指標をもとにデータを削除する．"""
        pass



    def update_sample_stats(self, sample_info):

        # 辞書の作成（データセットのインデックス : self.bufferのインデックス）
        db2buff = {b['idx']: i for i, b in enumerate(self.buffer)}

        # 学習に使用したサンプルの情報を取り出す
        sample_index = sample_info['meta']['index'].detach().cpu()    # データセットのインデックス
        sample_label = sample_info['meta']['label']                   # データのラベル

        z = sample_info['feature'][:].detach()
        sample_features = F.normalize(z, p=2, dim=-1)

        def polyak_avg(val, avg, gamma):
            return (1 - gamma) * val + gamma * avg
        

        for i in range(len(sample_index)):
            db_idx = sample_index[i].item()         # データセットのインデックス
            if db_idx in db2buff:
                b = self.buffer[db2buff[db_idx]]    # バッファ
                
                if not b['seen']:
                    b['feature'] = sample_features[i]
                else:
                    b['feature'] = F.normalize(polyak_avg(
                        b['feature'], sample_features[i], self.gamma),
                                               p=2,
                                               dim=-1)
                
                b['label'] = sample_label[i]
                b['seen'] = True
            #print("b['label'] : ", b['label'])
                
        samples = [
            self.buffer[db2buff[idx]] for idx in sample_index.tolist() if idx in db2buff
        ]
        
        if not samples:
            return {}
        else:
            return tensorize_buffer(samples)
        





    def __iter__(self):
        
        # 学習途中かどうかの確認．学習途中の記録がなければ各変数を初期化
        if not self.init_from_ckpt:
            self.db_head = 0
            self.num_batches_seen = 0
            self.num_batches_yielded = 0
            self.batch_history = deque(maxlen=128)
        

        # バッファサイズが全データ数よりも少ないかを確認
        assert self.buffer_size <= len(self.all_indices)


        # len(batch_sampler) を超えるまでミニバッチを作成して学習をする．
        while self.num_batches_yielded < len(self):

            # self.bufferにデータを追加
            indices_to_add = self.add_to_buffer(self.batch_size)
            self.indices_to_add = indices_to_add

            # バッファ内にデータが一定量たまるまで学習を行わずデータを溜め続ける
            if len(self.buffer) < self.buffer_size:
                # self.num_batches_yielded += 1
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


