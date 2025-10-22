
import time


import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


from utils import AverageMeter, WindowAverageMeter, ProgressMeter, adjust_learning_rate




def train_minred(model, model2, criterions, optimizer, trainloader, cfg, epoch, ckpt_manager=None, writer=None, scaler=None):


    # 学習状況を記録するための meter
    batch_time = WindowAverageMeter('Time', fmt=':6.3f')
    data_time = WindowAverageMeter('Data', fmt=':6.3f')
    mcc_losses = AverageMeter('MCCLoss', ':.4e')
    tcr_losses = AverageMeter('TCRLoss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    lr_meter = AverageMeter('LR', ':.4e')
    buff_meters = []
    if cfg.continual.buffer_type in ["minred"]:
        num_seen = AverageMeter('#Seen', ':6.3f')
        num_seen_max = AverageMeter('#Seen Max', ':6.3f')
        similarity = AverageMeter('Buffer Sim', ':6.3f')
        neig_similarity = AverageMeter('Buffer Neig Sim', ':6.3f')
        buff_meters = [num_seen, num_seen_max, similarity,
        neig_similarity]
    progress = ProgressMeter(len(trainloader), [batch_time, data_time, lr_meter] + buff_meters + [losses] + [mcc_losses] + [tcr_losses],
                             prefix="Epoch: [{}]".format(epoch),
                             tbwriter=writer)


    # model を trainモード，model2 を evalモード に変更
    model.train()
    model2.eval()

    # criterions の分解
    criterion = criterions["cos"]

    # DDP 環境
    local_rank = cfg.ddp.local_rank
    device = torch.device(f"cuda:{local_rank}")


    # amp の使用状況
    use_amp = bool(getattr(cfg, "amp", None) and cfg.amp.use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and str(cfg.amp.dtype).lower() == "bf16") else torch.float16


    print("len(trainloader): ", len(trainloader))

    end = time.time()
    for idx, data in enumerate(trainloader):

        # 学習済みのバッチ数をカウント
        batch_i = trainloader.batch_sampler.advance_batches_seen()
        
        # 学習率の調整
        effective_epoch = epoch + (batch_i / len(trainloader))
        lr = adjust_learning_rate(optimizer,
                                  effective_epoch,
                                  cfg,
                                  epoch_size=len(trainloader))
        lr_meter.update(lr)


        # 現在のタスクidを確認
        taskid = trainloader.batch_sampler.return_taskid()

        # 画像の用意
        images = data['input']
        meta = data["meta"]
        labels = meta["label"]
        data_time.update(time.time() - end)

        # gpu に配置
        if torch.cuda.is_available():
            images[0] = images[0].to(device, non_blocking=True)
            images[1] = images[1].to(device, non_blocking=True)
        


        # forward 処理と損失計算
        if use_amp:
            with autocast(dtype=amp_dtype): 
                p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                losses.update(loss.item(), images.size(0))

        else:
            p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            losses.update(loss.item(), images.size(0))



        # ==========
        # backward + step (AMP 対応)
        # ==========
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # クリッピングが必要ならここで
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()


        with torch.no_grad():
            
            # z1 と z2 の平均特徴を計算
            z_avg = None
            assert False
            data['feature'] = torch.stack((z1, z2), 1).detach()
            
            if cfg.continual.buffer_type in ["minred"]:
                stats = trainloader.batch_sampler.update_sample_stats(data)

                if 'num_seen' in stats:
                    num_seen.update(stats['num_seen'].float().mean().item(),
                                    stats['num_seen'].shape[0])
                    num_seen_max.update(stats['num_seen'].float().max().item(),
                                        stats['num_seen'].shape[0])
                if 'similarity' in stats:
                    similarity.update(stats['similarity'].float().mean().item(),
                                      stats['similarity'].shape[0])
                if 'neighbor_similarity' in stats:
                    neig_similarity.update(
                        stats['neighbor_similarity'].float().mean().item(),
                        stats['neighbor_similarity'].shape[0])














