
import os
import sys
import csv
import time
import math
import random
import numpy as np


import torch
from torchvision.utils import save_image
import torch.distributed as dist


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
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
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
        print("ckpt_fname: ", ckpt_fname)
        start_epoch = 0

        if os.path.isfile(ckpt_fname):
            checkpoint = torch.load(ckpt_fname, map_location='cpu')

            # Load state dict
            for k in self.modules:
                print("resume k: ", k)
                self.modules[k].load_state_dict(checkpoint[k])
            start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})".format(
                ckpt_fname, checkpoint['epoch']))
        
        return start_epoch

    # 処理時間を基にチェックポイントを保存する関数
    def timed_checkpoint(self, save_dict=None, taskid=None):
        
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
    def midway_epoch_checkpoint(self, epoch, batch_i, save_dict=None, taskid=None):
        if ((batch_i + 1) / float(self.epoch_size) % self.save_freq) < (batch_i / float(self.epoch_size) % self.save_freq):
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_{:010.4f}.pth')
            ckpt_fname = ckpt_fname.format(epoch + batch_i / float(self.epoch_size))


            # csvファイルに対応表を記録する
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            if not distributed or (distributed and torch.distributed.get_rank() == 0):
                self.write_model_taskid(ckpt_fname, taskid)

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, is_best=False, filename=ckpt_fname)
                ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_latest.pth')
                save_checkpoint(state, is_best=False, filename=ckpt_fname)
                print("モデルのパラメータなどを保存しました．{}".format(ckpt_fname))

    
    def create_state_dict(self, save_dict):
        state = {k: self.modules[k].state_dict() for k in self.modules}
        if save_dict is not None:
            state.update(save_dict)
        return state
    

    def write_model_taskid(self, ckpt_fname, taskid):

        # {self.ckpt_dir}/a_ckpt_taskid.csv に各モデルが何のタスクを学習中かの対応を書き込む
        file_path = f"{self.ckpt_dir}/a_ckpt_taskid.csv"
        
        if not os.path.isfile(file_path):
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # ヘッダー行を定義（必要に応じて適宜変更）
                header = ["taskid", "ckpt_fname"]
                writer.writerow(header)

        # CSV に実際のデータを追加記録する
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [taskid, ckpt_fname]
            writer.writerow(row)


    
    
    
    
    # ------------------------------------------
    # main.py と train() から呼び出される関数
    # ------------------------------------------
    def checkpoint(self, epoch, batch_i=None, save_dict=None, taskid=None):

        if batch_i is None:
            self.end_epoch_checkpoint(epoch, save_dict)
        else:
            if batch_i % 1 == 0:
                self.timed_checkpoint(save_dict)
            self.midway_epoch_checkpoint(epoch, batch_i, save_dict=save_dict, taskid=taskid)




# ==========================================
# 学習率の調整
# ==========================================
def adjust_learning_rate(optimizer, epoch, cfg, epoch_size=None):
    
    """Decay the learning rate based on schedule"""
    init_lr = cfg.optimizer.train.learning_rate
    if cfg.optimizer.train.scheduler_type == 'constant':
        cur_lr = init_lr

    elif cfg.optimizer.train.scheduler_type == 'cos':
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / cfg.optimizer.train.epochs))

    elif cfg.optimizer.train.scheduler_type == 'triangle':
        assert False
        # cfgの名前を修正していないので一旦エラーになるようにしました．必要ならcfgの内容を整えてから使用してください．
        # T = cfg.optimizer.lr_schedule.period
        # t = (epoch * epoch_size) % T
        # if t < T / 2:
        #     cur_lr = cfg.optimizer.train.learning_rate + t / (T / 2.) * (cfg.optimizer.lr_schedule.max_lr - cfg.optimizer.train.learning_rate)
        # else:
        #     cur_lr = cfg.optimizer.train.learning_rate + (T-t) / (T / 2.) * (cfg.optimizer.lr_schedule.max_lr - cfg.optimizer.train.learning_rate)

    else:
        raise ValueError('LR schedule unknown.')

    if cfg.optimizer.train.scheduler_exit_decay > 0:
        start_decay_epoch = cfg.optimizer.epochs * (1. - cfg.optimizer.lr_schedule.exit_decay)
        if epoch > start_decay_epoch:
            mult = 0.5 * (1. + math.cos(math.pi * (epoch - start_decay_epoch) / (cfg.optimizer.epochs - start_decay_epoch)))
            cur_lr = cur_lr * mult

    for param_group in optimizer.param_groups:
        
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    
    return cur_lr





# ==========================================
# 複数プロセス間で特徴を集約
# ==========================================
def concat_all_gather_no_grad(t: torch.Tensor) -> torch.Tensor:
    
    """他 rank の特徴を勾配なしで集約（デタッチ）。"""
    world_size = dist.get_world_size()
    
    if world_size == 1:
        return t
    
    tensors_gather = [torch.zeros_like(t) for _ in range(world_size)]
    with torch.no_grad():
        dist.all_gather(tensors_gather, t)  # 各rankの t を収集
    
    return torch.cat(tensors_gather, dim=0)




def concat_all_gather_keep_grad(x: torch.Tensor) -> torch.Tensor:
    """
    全 rank の特徴を集める。ただし **自 rank の x だけは detach しない**。
    これで勾配は自分のサンプルにだけ流れ、他 rank の特徴は定数として扱われる。
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x

    world = dist.get_world_size()
    rank = dist.get_rank()

    # いったん全 rank のテンソルを detatch で収集
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x.detach())             # ここは detach で OK

    # 自分の位置だけ「元の x（勾配あり）」に差し替える
    xs[rank] = x                                 # ← これが肝

    return torch.cat(xs, dim=0)




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














