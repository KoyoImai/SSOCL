
import os
import sys
import time
import random
import numpy as np


import torch
from torchvision.utils import save_image



# ==========================================
# 学習記録
# ==========================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', tbname=''):
        self.name = name
        self.tbname = tbname
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class WindowAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, k=250, fmt=':f', tbname=''):
        self.name = name
        self.tbname = tbname
        self.fmt = fmt
        self.k = k
        self.reset()

    def reset(self):
        from collections import deque
        self.vals = deque(maxlen=self.k)
        self.counts = deque(maxlen=self.k)
        self.val = 0
        self.avg = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.counts.append(n)
        self.val = val
        self.avg = sum([v * c for v, c in zip(self.vals, self.counts)]) / sum(
            self.counts)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", tbwriter=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.tbwriter = tbwriter

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        sys.stdout.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def tbwrite(self, batch):
        if self.tbwriter is None:
            return
        scalar_dict = self.tb_scalar_dict()
        for k, v in scalar_dict.items():
            self.tbwriter.add_scalar(k, v, batch)

    def tb_scalar_dict(self):
        out = {}
        for meter in self.meters:
            val = meter.avg
            if not meter.tbname:
                meter.tbname = meter.name
                tag = meter.tbname
                sclrval = val
                out[tag] = sclrval
        return out



        
# ==========================================
# seed値の固定
# ==========================================
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
# 学習途中の記録保存・再開用プログラム
# ==========================================
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    distributed = torch.distributed.is_available(
    ) and torch.distributed.is_initialized()
    if not distributed or (distributed and torch.distributed.get_rank() == 0):
        torch.save(state, filename)
        print("=> saved checkpoint '{}' (epoch {})".format(
            filename, state['epoch']))
        

class CheckpointManager:
    def __init__(self,
                 modules,
                 ckpt_dir,
                 epoch_size,
                 epochs,
                 save_freq=None,
                 save_freq_mints=None):
        
        self.modules = modules
        self.ckpt_dir = ckpt_dir
        self.epoch_size = epoch_size
        self.epochs = epochs
        self.save_freq = save_freq
        self.save_freq_mints = save_freq_mints

        self.time = time.time()
        self.distributed = torch.distributed.is_available(
        ) and torch.distributed.is_initialized()
        self.world_size = torch.distributed.get_world_size(
        ) if self.distributed else 1
        self.rank = torch.distributed.get_rank() if self.distributed else 0

        os.makedirs(os.path.join(self.ckpt_dir), exist_ok=True)
    

    def resume(self):

        ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_latest.pth')
        start_epoch = 0

        if os.path.isfile(ckpt_fname):
            checkpoint = torch.load(ckpt_fname, map_location='cpu')

            # Load state dict
            for k in self.modules:
                self.modules[k].load_state_dict(checkpoint[k])
            start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})".format(
                ckpt_fname, checkpoint['epoch']))
        
        return start_epoch

    # 処理時間を基にチェックポイントを保存する関数
    def timed_checkpoint(self, save_dict=None):
        
        # 経過時間の測定
        t = time.time() - self.time
        
        # 全プロセスの経過時間と集約
        t_all = [t for _ in range(self.world_size)]
        if self.world_size > 1:
            torch.distributed.all_gather_object(t_all, t)

        # 全プロセスで最も経過時間が短かったプロセスを基に現状を保存するか決定
        if min(t_all) > self.save_freq_mints * 60:

            # 時間の更新，保存パスの作成
            self.time = time.time()
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_latest.pth')

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, is_best=False, filename=ckpt_fname)
    

    # 学習の進捗を基にチェックポイントを保存する関数
    def midway_epoch_checkpoint(self, epoch, batch_i, save_dict=None):
        if ((batch_i + 1) / float(self.epoch_size) % self.save_freq) < (batch_i / float(self.epoch_size) % self.save_freq):
            ckpt_fname = os.path.join(self.ckpt_dir,
                                      'checkpoint_{:010.4f}.pth')
            ckpt_fname = ckpt_fname.format(epoch +
                                           batch_i / float(self.epoch_size))

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, is_best=False, filename=ckpt_fname)
                ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_latest.pth')
                save_checkpoint(state, is_best=False, filename=ckpt_fname)

    
    def create_state_dict(self, save_dict):
        state = {k: self.modules[k].state_dict() for k in self.modules}
        if save_dict is not None:
            state.update(save_dict)
        return state
    

    # ------------------------------------------
    # main.py と train() から呼び出される関数
    # ------------------------------------------
    def checkpoint(self, epoch, batch_i=None, save_dict=None):

        if batch_i is None:
            self.end_epoch_checkpoint(epoch, save_dict)
        else:
            if batch_i % 1 == 0:
                self.timed_checkpoint(save_dict)
            self.midway_epoch_checkpoint(epoch, batch_i, save_dict=save_dict)













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

