
import os
import hydra
import random
import builtins
import numpy as np


import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image


from utils import seed_everything, CheckpointManager
from models import make_model
from losses import make_criterion
from optimizers import make_optimizer
from augmentaions import make_transform
from dataloaders import make_dataset, make_sampler, make_batchsampler
from train import train





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
    # DDP 関連の処理を時効
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
        # print("dist.get_world_size(): ", dist.get_world_size())
    else:
        assert False


    # print 処理を master_node に限定する
    if cfg.ddp.use_ddp and (cfg.ddp.local_rank != 0):

        def print_pass(*args, **kwargs):
            pass
            
        builtins.print = print_pass
    
    if cfg.ddp.local_rank is not None:
        print("Use GPU: {} for training".format(cfg.ddp.local_rank))



    # ===========================================
    # tensorboard で記録するための準備
    # （一旦不要．必要なら後から実装）
    # ===========================================
    writer = None

    

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



    # ===========================================
    # 学習途中の記録があるなら読み込みを実行
    # プログラム全体の実装が完了したらここも実装
    # ===========================================
    # cfg.log.model_path
    modules = {
        'state_dict': model,
        'optimer': optimizer,
        'sampler': trainloader.batch_sampler
    }

    ckpt_manager = CheckpointManager(
        modules=modules,
        ckpt_dir=cfg.log.model_path,
        epoch_size=len(trainloader),
        epochs=cfg.optimizer.train.epochs,
        save_freq=cfg.log.save_freq,
        save_freq_mints=cfg.log.save_freq_mints
    )

    if cfg.log.resume:
        cfg.optimizer.train.start_epoch = ckpt_manager.resume()



    # ===========================================
    # 訓練を実行
    # ===========================================
    for epoch in range(cfg.optimizer.train.epochs):

        # ここは必要か？どうせ1エポックしか学習しないうえ，下手に学習順序変えると問題では？
        trainloader.batch_sampler.set_epoch(epoch=epoch)
        
        train(model=model, model2=model2, criterions=criterions, optimizer=optimizer,
              trainloader=trainloader, cfg=cfg, epoch=epoch, ckpt_manager=ckpt_manager, writer=writer)

    






if __name__ == '__main__':
    main()


