

import random
import numpy as np
import torch



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

    