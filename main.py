
import os
import hydra
import random
import numpy as np


import torch
import torch.nn as nn
import torch.distributed as dist



from utils import seed_everything
from models import make_model
from losses import make_criterion
from optimizers import make_optimizer
from augmentaions import make_transform
from dataloaders import make_dataset, make_sampler, make_batchsampler





def preparation(cfg):

    # データセット毎にタスク数・タスク毎のクラス数を決定
    # （現状不要）

    # 総タスク数
    # （現状不要）

    # モデルの保存，実験記録などの保存先パス
    if cfg.dataset.data_folder is None:
        cfg.dataset.data_folder = '~/data/'
    cfg.log.model_path = f'./logs/{cfg.method.name}/{cfg.log.name}/model/'      # modelの保存先
    cfg.log.explog_path = f'./logs/{cfg.method.name}/{cfg.log.name}/exp_log/'   # 実験記録の保存先
    cfg.log.mem_path = f'./logs/{cfg.method.name}/{cfg.log.name}/mem_log/'      # リプレイバッファ内の保存先
    cfg.log.result_path = f'./logs/{cfg.method.name}/{cfg.log.name}/result/'    # 結果の保存先

    # ディレクトリ作成
    if not os.path.isdir(cfg.log.model_path):
        os.makedirs(cfg.log.model_path)
    if not os.path.isdir(cfg.log.explog_path):
        os.makedirs(cfg.log.explog_path)
    if not os.path.isdir(cfg.log.mem_path):
        os.makedirs(cfg.log.mem_path)
    if not os.path.isdir(cfg.log.result_path):
        os.makedirs(cfg.log.result_path)





@hydra.main(config_path='configs/default/', config_name='default', version_base=None)
def main(cfg):

    # ===========================================
    # シード固定
    # ===========================================
    seed_everything(cfg.seed)


    # logの名前
    cfg.log.name = f"{cfg.log.base}_{cfg.method.name}_{cfg.continual.buffer_type}{cfg.continual.buffer_size}_{cfg.dataset.type}_seed{cfg.seed}_date{cfg.date}"


    # ===========================================
    # データローダ作成やディレクトリ作成などの前処理
    # ===========================================
    preparation(cfg)



    # ===========================================
    # DDP
    # ===========================================
    # DDP 使用の環境変数
    local_rank = int(os.environ["LOCAL_RANK"])
    use_ddp = local_rank != -1
    device = torch.device("cuda", local_rank if use_ddp else 0)
    
    cfg.ddp.local_rank = local_rank
    cfg.ddp.use_ddp = use_ddp
    cfg.ddp.world_size = os.environ['WORLD_SIZE']

    # DDP 使用
    if use_ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        print("dist.get_world_size(): ", dist.get_world_size())
    else:
        assert False


    # ===========================================
    # modelの作成
    # ===========================================
    model, model2 = make_model(cfg)


    # ===========================================
    # 損失関数の作成
    # ===========================================
    criterions = make_criterion(cfg)


    # ===========================================
    # Optimizerの作成
    # ===========================================
    optimizer = make_optimizer(cfg, model)
    # print("optimizer: ", optimizer)


    # ===========================================
    # データローダーの作成
    # ===========================================
    train_transform = make_transform(cfg)
    dataset = make_dataset(cfg, train_transform)
    sampler = make_sampler(cfg, dataset)
    batch_sampler = make_batchsampler(cfg, dataset, sampler)

    print("dataset: ", dataset)





if __name__ == '__main__':
    main()


