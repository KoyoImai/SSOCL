
import os
import hydra
import random
import numpy as np


import torch
import torch.nn as nn
import torch.distributed as dist



from utils import seed_everything
from models import make_model




@hydra.main(config_path='configs/default/', config_name='default', version_base=None)
def main(cfg):

    # ===========================================
    # シード固定
    # ===========================================
    seed_everything(cfg.seed)


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
    # 実験記録などを保存するディレクトリの作成
    # ===========================================
    cfg.log.model_dir = os.path.join(cfg.log.base_dir, cfg.log.name, cfg.log.model)
    cfg.log.result_dir = os.path.join(cfg.log.base_dir, cfg.log.name, cfg.log.result)
    os.makedirs(cfg.log.model_dir, exist_ok=True)
    os.makedirs(cfg.log.result_dir, exist_ok=True)



    # ===========================================
    # modelの作成
    # ===========================================
    model = make_model(cfg)



    # ===========================================
    # データローダーの作成
    # ===========================================




if __name__ == '__main__':
    main()


