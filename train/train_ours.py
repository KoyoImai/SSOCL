

import time


import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


from utils import AverageMeter, WindowAverageMeter, ProgressMeter, adjust_learning_rate, concat_all_gather_keep_grad



def train_ours(model, model2, criterions, optimizer, trainloader, cfg, epoch, ckpt_manager=None, writer=None, scaler=None):


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
    criterion_mcc = criterions["mcc"]
    criterion_tcr = criterions["tcr"]


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


        # 画像とラベルを獲得
        images = data["input"]
        meta = data["meta"]
        labels = meta["label"]

        # マルチクロップ画像の連結
        images = torch.cat(images, dim=0)
        data_time.update(time.time() - end)

        # gpuに配置
        if torch.cuda.is_available():
            images = images.to(device, non_blocking=True)
        

        # model の forward 処理
        encoded, feature, z_proj = model(images)
        # print("encoded.shape: ", encoded.shape)    # encoded.shape:  torch.Size([320, 2048])
        # print("feature.shape: ", feature.shape)    # feature.shape:  torch.Size([320, 4096])
        # print("z_proj.shape: ", z_proj.shape)      # z_proj.shape:  torch.Size([320, 1024])

        z_proj_concat = concat_all_gather_keep_grad(z_proj)  # [B_global * V, D]
        # print("z_proj.shape: ", z_proj.shape)       # z_proj.shape:  torch.Size([1280, 1024])




        # ==========================================
        # この部分は amp 非対応なので一旦コメントアウト
        # ==========================================
        # # 特徴量平均計算
        # z_list = z_proj.chunk(cfg.method.num_crops, dim=0)
        # z_avg = chunk_avg(z_proj, cfg.method.num_crops)

        # # MCC損失とTCR損失の計算
        # loss_mcc = criterion_mcc(z_list)
        # loss_tcr = cal_TCR(z_proj, criterion_tcr, cfg.method.num_crops)
        # mcc_losses.update(loss_mcc.item(), images.size(0))
        # tcr_losses.update(loss_tcr.item(), images.size(0))


        # # 損失の合算
        # loss = cfg.method.lambda_mcc * loss_mcc + cfg.method.lambda_tcr * loss_tcr
        # losses.update(loss.item(), images.size(0))


        if use_amp:
            with autocast(dtype=amp_dtype):
                z_list = z_proj_concat.chunk(cfg.method.num_crops, dim=0)
                z_avg = chunk_avg(z_proj, cfg.method.num_crops)

                loss_mcc = criterion_mcc(z_list)
                loss_tcr = cal_TCR(z_proj, criterion_tcr, cfg.method.num_crops)
                mcc_losses.update(loss_mcc.item(), images.size(0))
                tcr_losses.update(loss_tcr.item(), images.size(0))

                loss = cfg.method.lambda_mcc * loss_mcc + cfg.method.lambda_tcr * loss_tcr
                losses.update(loss.item(), images.size(0))

        else:
            z_list = z_proj_concat.chunk(cfg.method.num_crops, dim=0)
            z_avg = chunk_avg(z_proj, cfg.method.num_crops)

            loss_mcc = criterion_mcc(z_list)
            loss_tcr = cal_TCR(z_proj, criterion_tcr, cfg.method.num_crops)
            mcc_losses.update(loss_mcc.item(), images.size(0))
            tcr_losses.update(loss_tcr.item(), images.size(0))

            loss = cfg.method.lambda_mcc * loss_mcc + cfg.method.lambda_tcr * loss_tcr
            losses.update(loss.item(), images.size(0))

        
        
        
        # ==========================================
        # この部分は amp 非対応なので一旦コメントアウト
        # ==========================================
        # # 最適化ステップ
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()


        # ==========
        # backward + step (AMP 対応)
        # ==========
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()

            # 勾配クリッピングがある場合（例）：unscale 後に行う
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # クリッピングが必要ならここで
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()


        # バッファ内のサンプルの情報を更新する
        with torch.no_grad():

            # 平均特徴をバッファ内のサンプル情報として使用するため
            data["feature"] = z_avg.detach()

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
                    
            else:  
                assert False



        # 後から分析可能にするため，学習途中のモデルを一定間隔で保存する
        if ckpt_manager is not None:
            ckpt_manager.checkpoint(epoch=epoch,
                                    batch_i=batch_i,
                                    taskid=taskid,
                                    save_dict={
                                        'epoch': epoch,
                                        'batch_i': batch_i,
                                        'arch': cfg.model.type,
                                    })


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # 学習状況の表示
        if batch_i % cfg.log.print_freq == 0:
            tb_step = (epoch * len(trainloader.dataset) // cfg.optimizer.train.batch_size + batch_i * int(cfg.ddp.world_size))
            progress.display(batch_i)
            progress.tbwrite(tb_step)





def chunk_avg(x, n_chunks=2, normalize=False):
    x_list = x.chunk(n_chunks, dim=0)
    x = torch.stack(x_list, dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0), dim=1)
    
    
def cal_TCR(z, criterion, num_patches):
    
    z_list = z.chunk(num_patches, dim=0)
    # print("z_list[0].shape : ", z_list[0].shape)   # torch.Size([200, 1024])
    # assert False
    
    loss = 0
    for i in range(num_patches):
        loss += criterion(z_list[i])
    loss = loss / num_patches
    return loss






