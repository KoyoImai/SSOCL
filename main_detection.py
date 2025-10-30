
import os
import hydra

import torch
import torch.optim as optim


from models import make_backbone, load_pretrained_resnet_backbone
from models.detection import build_detection_model


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
        lr=cfg.detection.lr,
        momentum=cfg.detection.momentum,
        weight_decay=cfg.detection.weight_decay,
    )





    assert False







if __name__ == "__main__":
    main()



