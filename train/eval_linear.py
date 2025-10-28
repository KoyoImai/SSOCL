
import sys
import csv
import time
from itertools import zip_longest

import torch

from utils import AverageMeter, ProgressMeter


import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Possibly corrupt EXIF data.*",
    category=UserWarning,
    module=r"PIL\.TiffImagePlugin"
)

warnings.filterwarnings(
    "ignore",
    message=r"(?:Possibly corrupt|Corrupt) EXIF data.*",
    category=UserWarning,
    module=r"PIL\.(?:TiffImagePlugin|JpegImagePlugin)"
)



def save_all_stats_csv(all_stats, path):
    """
    all_stats = {
        'acc1': float or 0-dim tensor,
        'acc5': float or 0-dim tensor,
        'preds': 1D list/ndarray/tensor,
        'targets': 1D list/ndarray/tensor
    }
    """
    def to_scalar(x):
        try:
            # works for Python float/int and 0-dim tensors/ndarrays
            return float(x)
        except (TypeError, ValueError):
            try:
                return float(x.item())
            except Exception:
                return ""

    def to_list(x):
        # supports list/tuple/ndarray/torch.Tensor
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            x = x.detach().cpu()
        if hasattr(x, "tolist"):
            return x.tolist()
        return list(x)

    acc1 = to_scalar(all_stats.get("acc1", ""))
    acc5 = to_scalar(all_stats.get("acc5", ""))

    preds   = to_list(all_stats.get("preds", []))
    targets = to_list(all_stats.get("targets", []))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "pred", "target", "acc1", "acc5"])
        for i, (p, t) in enumerate(zip_longest(preds, targets, fillvalue="")):
            # unwrap possible 0-dim tensors for p/t
            if hasattr(p, "item") and callable(p.item): 
                try: p = p.item()
                except Exception: pass
            if hasattr(t, "item") and callable(t.item):
                try: t = t.item()
                except Exception: pass
            w.writerow([i, p, t, acc1, acc5])



def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def train_linear(model, classifier, criterion, optimizer, trainloader, valloader, epoch, scaler, writer, cfg):

    # modelをtrainモード，classifierをevalモードに変更
    model.eval()
    classifier.eval()

    all_stats = {}
    targets = []
    preds = []


    batch_time = AverageMeter('Time', ':6.3f', tbname='val/time')
    losses = AverageMeter('Loss', ':.4e', tbname='val/loss')
    top1 = AverageMeter('Acc@1', ':6.2f', tbname='val/top1')
    top5 = AverageMeter('Acc@5', ':6.2f', tbname='val/top5')
    progress = ProgressMeter(len(valloader), [batch_time, losses, top1, top5],
                             prefix='Test: ',
                             tbwriter=writer)
    

    # amp の使用状況
    use_amp = bool(getattr(cfg, "amp", None) and cfg.amp.use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and str(cfg.amp.dtype).lower() == "bf16") else torch.float16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(trainloader):

            images = data["input"]
            labels = data["target"]

            if torch.cuda.is_available():
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                features = model.encoder(images)
                # print("features.shape: ", features.shape)   # features.shape:  torch.Size([1024, 2048])

            outputs = classifier(features)
            # print("outputs.shape: ", outputs.shape)         # outputs.shape:  torch.Size([1024, 13971])

            loss = criterion(outputs, labels)
            # print("loss: ", loss)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if idx % cfg.log.print_freq == 0:
                progress.display(idx)
                sys.stdout.flush()


        progress.sync_distributed()
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

        progress.tbwrite(0)
        all_stats['acc1'] = top1.avg
        all_stats['acc5'] = top5.avg
        all_stats['preds'] = preds
        all_stats['targets'] = targets

    save_all_stats_csv(all_stats, "results.csv")

    return all_stats




