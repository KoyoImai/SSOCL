
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
                 rank,
                ) -> None:
        
        super().__init__(buffer_size, repeat, dataset, sampler, batch_size, rank)


    def update_sample_stats(self, sample_info):

        # 辞書の作成（データセットのインデックス : self.bufferのインデックス）
        db2buff = {b['idx']: i for i, b in enumerate(self.buffer)}

        # 学習に使用したサンプルの情報を取り出す
        sample_index = sample_info['meta']['index'].detach().cpu()    # データセットのインデックス
        sample_label = sample_info['meta']['label']                   # データのラベル

        z = sample_info['feature'][:].detach()
        sample_features = F.normalize(z, p=2, dim=-1)

        
        # 指数移動平均を計算するための関数
        def polyak_avg(val, avg, gamma):

            if avg.device != val.device:
                avg = avg.to(val.device, non_blocking=True)
            
            return (1 - gamma) * val + gamma * avg
        
        # サンプルを順番に処理を実行
        for i in range(len(sample_index)):
            db_idx = sample_index[i].item()         # データセットのインデックス
            if db_idx in db2buff:
                b = self.buffer[db2buff[db_idx]]    # バッファ
                
                if not b['seen']:
                    b['feature'] = sample_features[i].detach().to("cpu")
                else:
                    b['feature'] = F.normalize(polyak_avg(
                        b['feature'], sample_features[i].detach().to("cpu"), self.gamma), p=2, dim=-1)
                
                b['label'] = sample_label[i]
                b['num_seen'] += 1
                b['seen'] = True
            #print("b['label'] : ", b['label'])
                
        samples = [
            self.buffer[db2buff[idx]] for idx in sample_index.tolist() if idx in db2buff
        ]
        
        if not samples:
            return {}
        else:
            return tensorize_buffer(samples)
        

    def _resize_buffer(self, n) -> None:
        """buffer_size を超えている場合，何かしらの指標をもとにデータを削除する．"""

        n2rm = len(self.buffer) - n
        if n2rm <= 0:
            return

        # 類似度が高いデータのインデックスを返す
        def max_coverage_reductioin(x, n2rm):
            
            # 各データの特徴量の類似度を計算
            sim = (torch.einsum('ad,bd->ab', x, x) + 1) / 2

            # 同じデータ同士の類似度をマスクする
            sim.fill_diagonal_(-10.)

            # 削除データのインデックスを保存するリストの初期化
            idx2rm = []

            for i in range(n2rm):
                neig_sim = sim.max(dim=1)[0]
                most_similar_idx = torch.argmax(neig_sim)
                idx2rm += [most_similar_idx.item()]
                sim.index_fill_(0, most_similar_idx, -10.)
                sim.index_fill_(1, most_similar_idx, -10.)

            # idx2rmは，bufferにおける削除するデータのインデックスを格納している
            return idx2rm

        # self.bufferからインデックスとバッファの内容を取り出す
        buffer = [(b, i) for i, b in enumerate(self.buffer) if b['seen']]

        # 削除対象のデータ数が少ない場合，lifespanの値を元に削除データを決定
        if len(buffer) < 2 * n2rm:
            lifespans = [b['lifespan'] for b in self.buffer]
            idx2rm = torch.tensor(lifespans).argsort(
                descending=True)[:n2rm].tolist()
        
        else:

            # バッファに保存しているデータの特徴量を取り出して結合
            feats = torch.stack([b['feature'] for b, i in buffer], 0)
            
            # 冗長なデータのbufferにおけるインデックスを獲得
            idx2rm = max_coverage_reductioin(feats, n2rm)

            # 削除する冗長なデータのself.bufferにおけるインデックスを獲得
            idx2rm = [buffer[i][1] for i in idx2rm]


        # idx2rm（削除するデータのbufferでのインデックス）を整頓
        idx2rm = set(idx2rm)

        # バッファからデータを削除
        self.buffer = [b for i, b in enumerate(self.buffer) if i not in idx2rm]

        # Recompute nearest neighbor similarity for tracking
        if any(b['seen'] for b in self.buffer):
            feats = torch.stack(
                [b['feature'] for b in self.buffer if b['seen']], 0)
            feats = feats.cuda() if torch.cuda.is_available() else feats
            feats_sim = torch.einsum('ad,bd->ab', feats, feats)
            neig_sim = torch.topk(feats_sim, k=2, dim=-1,
                                  sorted=False)[0][:, 1:].mean(dim=1).cpu()
            i = 0
            for b in self.buffer:
                if b['seen']:
                    b['neighbor_similarity'] = neig_sim[i]
                    i += 1



    def __iter__(self):
        
        # 学習途中かどうかの確認．学習途中の記録がなければ各変数を初期化
        if not self.init_from_ckpt:
            self.db_head = 0
            self.num_batches_seen = 0
            self.num_batches_yielded = 0
            self.batch_history = deque(maxlen=128)


        # モデルが未確認のミニバッチを再度渡す
        for i in range(self.num_batches_yielded - self.num_batches_seen, 0, -1):
            yield self.batch_history[-i]
        

        # バッファサイズが全データ数よりも少ないかを確認
        assert self.buffer_size <= len(self.all_indices)


        # len(batch_sampler) を超えるまでミニバッチを作成して学習をする．
        while self.num_batches_yielded < len(self):

            # self.bufferにデータを追加
            indices_to_add = self.add_to_buffer(self.batch_size)
            self.indices_to_add = indices_to_add

            # バッファ内にデータが一定量たまるまで学習を行わずデータを溜め続ける
            # if len(self.buffer) < int(self.buffer_size // 2):
            if len(self.buffer) < self.buffer_size:
                continue

            # バッファからデータを削除
            self._resize_buffer(self.buffer_size)

            # repeat 回繰り返してミニバッチを作成
            for j in range(self.repeat):

                # ミニバッチの作成
                batch_idx = self.make_batch()
                # print("batch_idx: ", batch_idx)

                self.num_batches_yielded += 1
                self.batch_history += [batch_idx]

                yield batch_idx
        
        # 学習途中からの再開フラグを False に変更
        self.init_from_ckpt = False
            

    def __len__(self) -> int:
        return len(self.sampler) * self.repeat // self.batch_size


