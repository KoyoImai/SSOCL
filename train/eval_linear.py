
import os
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




def write_csv(value, path, file_name, task, dataset, epoch, ckpt=None):

    # ファイルパスを生成
    if dataset == "imagenet21k":
        file_path = f"{path}/{file_name}_task{task}.csv"
    else:
        file_path = f"{path}/{file_name}_task{task}_{dataset}.csv"

    # ファイルが存在しなければ新規作成、かつヘッダー行を記入する
    # value がリストの場合は、ヘッダーの値部分は要素数に合わせて "value_1", "value_2", ... とする例
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # ヘッダー行を定義（必要に応じて適宜変更）
            if isinstance(value, list):
                header = ["ckpt"], ["task"] + ["epoch"] + [f"task_{i+1}" for i in range(len(value))]
            else:
                header = ["ckpt", "task", "epoch", "value"]
            writer.writerow(header)

    # CSV に実際のデータを追加記録する
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if isinstance(value, list):
            row = [ckpt] + [task] + [epoch] + value
        else:
            row = [ckpt, task, epoch, value]
        writer.writerow(row)


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



def eval_linear(model, classifier, criterion, optimizer, trainloader, valloader, epoch, scaler, writer, cfg):

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

    # チェックポイント
    ckpt = cfg.linear.ckpt


    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(valloader):

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


        # progress.sync_distributed()
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

        progress.tbwrite(0)
        all_stats['acc1'] = top1.avg
        all_stats['acc5'] = top5.avg


    if cfg.linear.task_id == []:
        write_csv(top1.avg.item(), cfg.log.result_path, file_name="top1_acc", dataset=cfg.linear.dataset, task="all", epoch=epoch, ckpt=ckpt)
        write_csv(top5.avg.item(), cfg.log.result_path, file_name="top5_acc", dataset=cfg.linear.dataset, task="all", epoch=epoch, ckpt=ckpt)
    else:
        write_csv(top1.avg.item(), cfg.log.result_path, file_name="top1_acc", dataset=cfg.linear.dataset, task=cfg.linear.task_id, epoch=epoch, ckpt=ckpt)
        write_csv(top5.avg.item(), cfg.log.result_path, file_name="top5_acc", dataset=cfg.linear.dataset, task=cfg.linear.task_id, epoch=epoch, ckpt=ckpt)

    return top1.avg.item()




