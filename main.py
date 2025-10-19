
import os
import hydra
import random
import numpy as np


import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image


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


def denorm(x, mean, std):
    x = x.detach().cpu().clone()
    for c, m, s in zip(range(x.shape[0]), mean, std):
        x[c] = x[c] * s + m
    return x.clamp(0, 1)


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
    trainloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=cfg.workers, pin_memory=True)

    # for data in trainloader:

    #     mean=(0.5, 0.5, 0.5)
    #     std=(0.5, 0.5, 0.5)

    #     images = data["input"]
    #     meta = data["meta"]

    #     images = torch.cat(images, dim=0)

    #     # if cfg.ddp.local_rank == 0:
    #     #     print("images.shape: ", images.shape)   # images.shape:  torch.Size([500, 3, 224, 224])
    #     # images[0:batch_size] が ミニバッチに対するデータ拡張の一つめ，images[batch_size:2*batch_size] はデータ拡張の二つめ
        


    #     # images[0]を保存
    #     img0 = denorm(images[0], mean, std)
    #     os.makedirs("outputs/vis", exist_ok=True)
    #     save_image(img0, "outputs/vis/img0_denorm.png")

    #     # images[1]を保存
    #     img1 = denorm(images[1], mean, std)
    #     os.makedirs("outputs/vis", exist_ok=True)
    #     save_image(img1, "outputs/vis/img1_denorm.png")

    #     # images[20]を保存
    #     img20 = denorm(images[20], mean, std)
    #     os.makedirs("outputs/vis", exist_ok=True)
    #     save_image(img20, "outputs/vis/img20_denorm.png")

    #     # images[25]を保存
    #     img25 = denorm(images[25], mean, std)
    #     os.makedirs("outputs/vis", exist_ok=True)
    #     save_image(img25, "outputs/vis/img25_denorm.png")








if __name__ == '__main__':
    main()


