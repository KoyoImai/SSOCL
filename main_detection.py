
import os
import hydra
from typing import Dict, Iterable, List, Optional, Tuple
import bisect

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


from models import make_backbone, load_pretrained_resnet_backbone
from models.detection import build_detection_model
from augmentaions import make_detection_augmentation
from dataloaders import make_detection_dataset
from train import detector_train, detector_eval

from torch.optim.lr_scheduler import _LRScheduler


from utils import seed_everything




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




class WarmupMultiStepLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of increasing integers")
        if warmup_method not in ("constant", "linear"):
            raise ValueError("Warmup method must be 'constant' or 'linear'")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def _get_warmup_factor_at_iter(self, iteration):
        if iteration >= self.warmup_iters:
            return 1.0

        if self.warmup_method == "constant":
            return self.warmup_factor
        if self.warmup_method == "linear":
            alpha = iteration / self.warmup_iters
            return self.warmup_factor * (1 - alpha) + alpha

        raise ValueError("Unknown warmup method: {}".format(self.warmup_method))

    def get_lr(self):
        warmup_factor = self._get_warmup_factor_at_iter(self.last_epoch)
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect.bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]



@hydra.main(config_path="configs/detection", config_name="default", version_base=None)
def main(cfg):

    # ===========================================
    # シード値固定
    # ===========================================
    seed_everything(cfg.seed)

    # logの名前を決定
    cfg.log.name = f"{cfg.log.base}_{cfg.method.name}_{cfg.continual.buffer_type}{cfg.continual.buffer_size}_{cfg.dataset.type}_seed{cfg.seed}_date{cfg.date}"

    device = torch.device("cuda")


    # ===========================================
    # データローダ作成やディレクトリ作成などの前処理
    # ===========================================
    preparation(cfg)

    # ===========================================
    # backbone model の作成
    # ===========================================
    backbone = make_backbone(cfg)
    backbone, _ = load_pretrained_resnet_backbone(cfg, backbone)
    # print("backbone: ", backbone)

    
    # ===========================================
    # detection model の作成
    # ===========================================
    model = build_detection_model(cfg, backbone)
    # print("model: ", model)

    if torch.cuda.is_available():
        model.to(device)

    
    # ===========================================
    # Optimizer の作成
    # ===========================================
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=cfg.detection.train.learning_rate,
        momentum=cfg.detection.train.momentum,
        weight_decay=cfg.detection.train.weight_decay,
    )

    # ===========================================
    # Scheduler の作成
    # ===========================================
    # lr_scheduler = MultiStepLR(optimizer, milestones=cfg.detection.train.milestones, gamma=cfg.detection.train.gamma)
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=cfg.detection.train.milestones,
        gamma=cfg.detection.train.gamma,
        warmup_factor=cfg.detection.train.warmup_factor,
        warmup_iters=cfg.detection.train.warmup_iters,
        warmup_method=cfg.detection.train.warmup_method,
    )


    # ===========================================
    # Dataloader の作成
    # ===========================================
    train_augmentation, test_augmentation = make_detection_augmentation(cfg)
    train_dataset, test_dataset = make_detection_dataset(cfg, train_augmentation, test_augmentation)

    def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
        images, targets = zip(*batch)
        return list(images), list(targets)
    
    # 354864

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.detection.train.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.detection.eval.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )




    # ===========================================
    # amp の作成
    # ===========================================
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.detection.use_amp)


    # =========================
    # 訓練と評価
    # =========================
    best_map = 0.0
    for epoch in range(cfg.detection.train.epochs):

        # 学習の実行
        # detector_train(model, train_loader, cfg.detection.dataset.train_folder, optimizer, lr_scheduler, device, scaler, epoch, cfg.log.print_freq, cfg)

        # 評価の実行
        metrics = detector_eval(model, test_loader, cfg.detection.dataset.test_folder, optimizer, lr_scheduler, device, scaler, epoch, cfg.log.print_freq, cfg)
        map_value = metrics.get("map", 0.0)
        print(f"Validation mAP: {map_value:.4f}")




    







if __name__ == "__main__":
    main()



