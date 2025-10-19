
import os
import random
import numpy as np


import torch
from torchvision.utils import save_image



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


        

def seed_everything(seed):
    # Python 内部のハッシュシードを固定（辞書等の再現性に寄与）
    # os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Python 標準の乱数生成器のシード固定
    random.seed(seed)
    
    # NumPy の乱数生成器のシード固定
    np.random.seed(seed)
    
    # PyTorch のシード固定
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # マルチGPU対応の場合
    # Deterministic モードの有効化（PyTorch の一部非決定的な処理の回避）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    




# ==========================================
# ここから下は debug 用のプログラム
# ==========================================
def denorm(x, mean, std):
    x = x.detach().cpu().clone()
    for c, m, s in zip(range(x.shape[0]), mean, std):
        x[c] = x[c] * s + m
    return x.clamp(0, 1)


def util_dataloader(model, trainloader, cfg):

    for data in trainloader:

        local_rank = cfg.ddp.local_rank
        device = torch.device(f"cuda:{local_rank}")

        mean=(0.5, 0.5, 0.5)
        std=(0.5, 0.5, 0.5)

        images = data["input"]
        meta = data["meta"]

        images = torch.cat(images, dim=0)

        if torch.cuda.is_available():
            images = images.to(device, non_blocking=True)

        # if cfg.ddp.local_rank == 0:
        #     print("images.shape: ", images.shape)   # images.shape:  torch.Size([500, 3, 224, 224])
        # images[0:batch_size] が ミニバッチに対するデータ拡張の一つめ，images[batch_size:2*batch_size] はデータ拡張の二つめ
        
        # images[0]を保存
        img0 = denorm(images[0], mean, std)
        os.makedirs("outputs/vis", exist_ok=True)
        save_image(img0, "outputs/vis/img0_denorm.png")

        encoded, feature, z = model(images)
        print("encoded.shape: ", encoded.shape)
        print("feature.shape: ", feature.shape)
        print("z.shape: ", z.shape)

