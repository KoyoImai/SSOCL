
import sys
import time

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
    classifier.train()


    batch_time = AverageMeter('Time', ':6.3f', tbname='train/time')
    data_time = AverageMeter('Data', ':6.3f', tbname='train/datatime')
    losses = AverageMeter('Loss', ':.4e', tbname='train/loss')
    lr_meter = AverageMeter('LR', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f', tbname='train/top1')
    top5 = AverageMeter('Acc@5', ':6.2f', tbname='train/top5')
    progress = ProgressMeter(len(trainloader),
                             [batch_time, data_time, losses, lr_meter, top1, top5],
                             prefix="Epoch: [{}]".format(epoch),
                             tbwriter=writer)
    

    # amp の使用状況
    use_amp = bool(getattr(cfg, "amp", None) and cfg.amp.use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and str(cfg.amp.dtype).lower() == "bf16") else torch.float16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 現在の学習率
        current_lr = optimizer.param_groups[0]['lr']
        lr_meter.update(current_lr)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if idx % cfg.log.print_freq == 0:
            progress.display(idx)
            sys.stdout.flush()








