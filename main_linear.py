
import os
import hydra
import builtins


import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler



from models import make_model, make_classifier
from augmentaions import make_transform_eval
from dataloaders import make_dataset_eval
from optimizers import make_optimizer_eval
from losses import make_criterion_eval
from train import linear_train, adjust_learning_rate, linear_eval
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


def make_ddp(cfg):

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
    

# DDP の world_size に応じて学習ハイパラを調整
def setup_hypara(cfg):

    # バッチサイズを各プロセスで均等に分割
    cfg.optimizer.train.batch_size = int(cfg.optimizer.train.batch_size / int(cfg.ddp.world_size))


def make_amp(cfg):
    
    use_amp = bool(getattr(cfg, "amp", None) and cfg.amp.use_amp)
    use_bf16 = use_amp and (str(cfg.amp.dtype).lower() == "bf16")
    use_fp16 = use_amp and (str(cfg.amp.dtype).lower() == "fp16")
    scaler = None
    if use_fp16 and bool(getattr(cfg.amp, "grad_scaler", True)):
        scaler = GradScaler(enabled=True)
    # bf16 はスケーリング不要（数値範囲が広い）

    # 参考: TF32 を許可（速度重視、学習再現性は若干変わる場合あり）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True






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
    classifier = make_classifier(cfg)


    # ===========================================
    # Optimizer の作成
    # ===========================================
    optimizer = make_optimizer_eval(cfg, classifier)


    # ===========================================
    # 損失関数の作成
    # ===========================================
    criterion = make_criterion_eval(cfg)


    # ===========================================
    # 学習済みパラメータの読み込み
    # ===========================================
    # ---------- 観察対象（最初の1パラメータ）のスナップショット ----------
    # name, before = next(iter(model.state_dict().items()))
    # before = before.detach().cpu().clone()
    # print(f"[TARGET] {name}")
    # print("before[:5]:", before.flatten()[:5])


    ckpt_path = f"{cfg.log.model_path}/checkpoint_00001.0000.pth"
    # ckpt_path = f"{cfg.log.model_path}/checkpoint_00000.9600.pth"

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # print("checkpoint.keys(): ", checkpoint.keys())                                # checkpoint.keys():  dict_keys(['state_dict', 'optimizer', 'sampler', 'epoch', 'batch_i', 'arch'])
    # print("checkpoint['state_dict'].keys(): ", checkpoint['state_dict'].keys())
    # print("checkpoint['sampler'].keys(): ", checkpoint['sampler'].keys())          # checkpoint['sampler'].keys():  dict_keys(['buffer', 'db_head', 'num_batches_seen', 'num_batches_yielded', 'batch_history'])
    # print("checkpoint['sampler']['db_head']: ", checkpoint['sampler']['db_head'])
    # print("checkpoint['sampler']['buffer']: ", checkpoint['sampler']['buffer'])
    # print("checkpoint['sampler']['buffer']['idx']: ", checkpoint['sampler']['buffer']['idx'])
    # print("len(checkpoint['sampler']['buffer']['idx'][0]): ", len(checkpoint['sampler']['buffer']['idx'][0]))

    # # model の各パラメータの名称を変更し，module を取り除く
    state_dict = checkpoint["state_dict"]
    state_dict_wo_module = { (k[7:] if k.startswith("module") else k): v for k, v in state_dict.items() }
    model.load_state_dict(state_dict=state_dict_wo_module, strict=True)



    # # ---------- 読み込み後の同パラメータを確認 ----------
    # after = model.state_dict()[name].detach().cpu()
    # print("after[:5]: ", after.flatten()[:5])
    # print("changed?: ", not torch.equal(before, after))
    # print("max|diff|: ", (before - after).abs().max().item())



    # ===========================================
    # データローダーの作成
    # ===========================================
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError(
            "このスクリプトは GPU 前提です。CUDA/GPU が見えていません。"
        )

    train_transform, val_transform = make_transform_eval(cfg)
    train_dataset, val_dataset = make_dataset_eval(cfg, train_transform, val_transform)

    # trainloader = DataLoader(dataset=train_dataset,
    #                          batch_size=cfg.linear.train.batch_size,
    #                          shuffle=True,
    #                          num_workers=cfg.linear.num_workers,
    #                          pin_memory=True,
    #                          drop_last=True,
    #                          pin_memory_device='cuda' if use_cuda else 'cpu',
    #                          )

    # valloader = DataLoader(dataset=val_dataset,
    #                        batch_size=500,
    #                        shuffle=False,
    #                        num_workers=cfg.linear.num_workers,
    #                        pin_memory=True,
    #                        drop_last=False,
    #                        pin_memory_device='cuda' if use_cuda else 'cpu',
    #                        )
    trainloader = DataLoader(dataset=train_dataset,
                             batch_size=cfg.linear.train.batch_size,
                             shuffle=True,
                             num_workers=cfg.linear.num_workers,
                            #  pin_memory=True,
                             drop_last=True,
                             )

    valloader = DataLoader(dataset=val_dataset,
                           batch_size=500,
                           shuffle=False,
                           num_workers=cfg.linear.num_workers,
                           pin_memory=True,
                           drop_last=False,
                           )




    # =========================
    # AMP: GradScaler 準備
    # =========================
    scaler = make_amp(cfg)


    # ===========================================
    # tensorboard で記録するための準備
    # （一旦不要．必要なら後から実装）
    # ===========================================
    writer = None


    # =========================
    # 評価 
    # =========================
    best_acc = 0.0
    for epoch in range(cfg.linear.train.epochs):

        adjust_learning_rate(optimizer, epoch, cfg)

        linear_train(model=model, classifier=classifier, criterion=criterion, optimizer=optimizer,
                     trainloader=trainloader, valloader=valloader, epoch=epoch, scaler=scaler, writer=writer, cfg=cfg)

        top1_acc = linear_eval(model=model, classifier=classifier, criterion=criterion, optimizer=optimizer,
                               trainloader=trainloader, valloader=valloader, epoch=epoch, scaler=scaler, writer=writer, cfg=cfg)

        if top1_acc > best_acc:
            best_acc = top1_acc
            checkpoint_path = os.path.join(cfg.log.model_path, "best_linear.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved new best detection checkpoint to {checkpoint_path}")



if __name__ == '__main__':
    main()



