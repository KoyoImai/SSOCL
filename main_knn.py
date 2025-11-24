import os
import hydra
import builtins


import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler



from models import make_model, make_classifier
from augmentaions import make_transform_knn
from dataloaders import make_dataset_knn
from optimizers import make_optimizer_eval
from losses import make_criterion_eval
from train import knn_eval
from utils import seed_everything, CheckpointManager




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
    # シード値固定
    # ===========================================
    seed_everything(cfg.seed)

    # logの名前を決定
    cfg.log.name = f"{cfg.log.base}_{cfg.method.name}_{cfg.continual.buffer_type}{cfg.continual.buffer_size}_{cfg.dataset.type}_seed{cfg.seed}_date{cfg.date}"


    # # ===========================================
    # # DDP 関連の処理を実行
    # # （一旦なし．可能な限り1gpuで実行する）
    # # ===========================================
    # make_ddp(cfg)
    # setup_hypara(cfg)


    # ===========================================
    # データローダ作成やディレクトリ作成などの前処理
    # ===========================================
    preparation(cfg)

    
    # ===========================================
    # model と classifier の作成
    # ===========================================
    model, _ = make_model(cfg, use_ddp=False)


    # ===========================================
    # 学習済みパラメータの読み込み
    # ===========================================
    ckpt_path = f"{cfg.log.model_path}/{cfg.knn.ckpt}"
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # model の各パラメータの名称を変更し，module を取り除く
    state_dict = checkpoint["state_dict"]
    state_dict_wo_module = { (k[7:] if k.startswith("module") else k): v for k, v in state_dict.items() }
    model.load_state_dict(state_dict=state_dict_wo_module, strict=True)


    # ===========================================
    # データローダーの作成
    # ===========================================
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError(
            "このスクリプトは GPU 前提です。CUDA/GPU が見えていません。"
        )

    train_transform, val_transform = make_transform_knn(cfg)
    train_dataset, val_dataset = make_dataset_knn(cfg, train_transform, val_transform)

    trainloader = DataLoader(dataset=train_dataset,
                             batch_size=cfg.linear.train.batch_size,
                             shuffle=True,
                             num_workers=cfg.linear.num_workers,
                            #  pin_memory=True,
                             drop_last=False,
                             )

    valloader = DataLoader(dataset=val_dataset,
                           batch_size=500,
                           shuffle=False,
                           num_workers=cfg.linear.num_workers,
                        #    pin_memory=True,
                           drop_last=False,
                           )


    # ===========================================
    # tensorboard で記録するための準備
    # （一旦不要．必要なら後から実装）
    # ===========================================
    writer = None

    # knn分類による評価
    knn_eval(model=model, trainloader=trainloader, valloader=valloader, writer=writer, cfg=cfg)


if __name__ == '__main__':
    main()

