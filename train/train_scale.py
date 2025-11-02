

import time
import copy


import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


from utils import AverageMeter, WindowAverageMeter, ProgressMeter, adjust_learning_rate, concat_all_gather_keep_grad

torch.autograd.set_detect_anomaly(True)

def similarity_mask_old(feat_all, bsz, cfg, pos_pairs, device):
    """Calculate the pairwise similarity and the mask for contrastive learning
    Args:
        feat_all: all hidden features of shape [n_views * bsz, ...].
        bsz: int, batch size of input data (stacked streaming and memory samples)（合計のバッチサイズ）
        opt: arguments
        pos_pairs: averagemeter recording number of positive pairs
    Returns:
        contrast_mask: mask of shape [bsz, bsz]
    """
    #print(feat_all[0])
    #print(feat_all[1])
    
    feat_size = feat_all.size(0)
    n_views = int(feat_size / bsz)
    assert (n_views * bsz == feat_size), "Unmatch feature sizes and batch size!"

    # Compute the pairwise distance and similarity between each view
    # and add the similarity together for average
    simil_mat_avg = torch.zeros(bsz, bsz).to(device)
    mat_cnt = 0
    for i in range(n_views):
        for j in range(n_views):
            # feat_row and feat_col should be of size [bsz^2, bsz^2]
            #feat_row, feat_col = PairEnum(feat_all[i*bsz: (i+1)*bsz],
            #                              feat_all[j*bsz: (j+1)*bsz])
            #tmp_distance = -(((feat_row - feat_col) / temperature) ** 2.).sum(1)  # Euclidean distance
            # Note, all features are normalized
            if cfg.method.simil == 'kNN':  # euclidean distance
                simil_mat = 2 - 2 * torch.matmul(feat_all[i*bsz: (i+1)*bsz],
                                                 feat_all[j*bsz: (j+1)*bsz].T)
            elif cfg.method.simil == 'tSNE':  # tSNE similarity
                # compute euclidean distance pairs
                simil_mat = 2 - 2 * torch.matmul(feat_all[i*bsz: (i+1)*bsz],
                                                 feat_all[j*bsz: (j+1)*bsz].T)
                #print('\teuc dist', simil_mat * 1e4)
                tmp_distance = - torch.div(simil_mat, cfg.method.temp_tsne)
                tmp_distance = tmp_distance - 1000 * torch.eye(bsz).to(device)
                #print('\ttemp dist', tmp_distance * 1e4)
                simil_mat = 0.5 * torch.softmax(tmp_distance, 1) + 0.5 * torch.softmax(tmp_distance, 0)
                #print(torch.softmax(tmp_distance, 1))
                #print('simil_mat', simil_mat)
            else:
                raise ValueError(cfg.method.simil)

            # Add the new probability to the average probability
            simil_mat_avg = (mat_cnt * simil_mat_avg + simil_mat) / (mat_cnt + 1)
            mat_cnt += 1
    #print('simil_mat_avg', simil_mat_avg * 1e4)
    logits_mask = torch.scatter(
        torch.ones_like(simil_mat_avg),
        1,
        torch.arange(simil_mat_avg.size(0)).view(-1, 1).to(device),
        0
    )
    simil_max = simil_mat_avg[logits_mask.bool()].max()
    simil_mean = simil_mat_avg[logits_mask.bool()].mean()
    simil_min = simil_mat_avg[logits_mask.bool()].min()
    #print('prob_simil_avg: dim {}\tmax {}\tavg {}\tmin {}'.format(
    #    simil_mat_avg.shape[0], simil_max, simil_mean, simil_min))
    # Set diagonal of similarity matrix to ones
    masks = torch.eye(bsz).to(device)
    simil_mat_avg = simil_mat_avg * (1 - masks) + masks

    # mask out memory elements
    stream_mask = torch.zeros_like(simil_mat_avg).float().to(device)
    stream_mask[:cfg.optimizer.train.stream_batch_size, :cfg.optimizer.train.stream_batch_size] = 1
    simil_mat_avg = simil_mat_avg * stream_mask

    contrast_mask = torch.zeros_like(simil_mat_avg).float().to(device)
    if cfg.method.simil == 'tSNE':
        simil_thres = simil_mean + cfg.method.thres_ratio * (simil_max - simil_mean)
        # print(simil_thres)
        contrast_mask[simil_mat_avg > simil_thres] = 1
    elif cfg.method.simil == 'kNN':
        contrast_mask[:cfg.optimizer.train.batch_size, :cfg.optimizer.train.batch_size][
            simil_mat_avg[:cfg.optimizer.train.batch_size, :cfg.optimizer.train.batch_size] <
            torch.kthvalue(simil_mat_avg[:cfg.optimizer.train.batch_size, :cfg.optimizer.train.batch_size],
                           int(cfg.method.simil_thres), 1, True)[0]] = 1
        contrast_mask[:cfg.optimizer.train.batch_size, :cfg.optimizer.train.batch_size][
            simil_mat_avg[:cfg.optimizer.train.batch_size, :cfg.optimizer.train.batch_size] <
            torch.kthvalue(simil_mat_avg[:cfg.optimizer.train.batch_size, :cfg.optimizer.train.batch_size],
                           int(cfg.method.simil_thres), 0, True)[0]] = 1

    pos_pairs.update(contrast_mask.sum().item() / cfg.optimizer.train.batch_size, bsz)
    # print('Avg num of positive samples: {}'.format(pos_pairs.val))

    return contrast_mask




def train_scale(model, model2, criterions, optimizer, trainloader, cfg, epoch, ckpt_manager=None, writer=None, scaler=None):


    # 学習状況を記録するための meter
    batch_time = WindowAverageMeter('Time', fmt=':6.3f')
    data_time = WindowAverageMeter('Data', fmt=':6.3f')
    psc_losses = AverageMeter('MCCLoss', ':.4e')
    ird_losses = AverageMeter('TCRLoss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    pos_pairs = AverageMeter('PosPairs', fmt=':4.1f')
    lr_meter = AverageMeter('LR', ':.4e')
    buff_meters = []
    if cfg.continual.buffer_type in ["minred"]:
        num_seen = AverageMeter('#Seen', ':6.3f')
        num_seen_max = AverageMeter('#Seen Max', ':6.3f')
        similarity = AverageMeter('Buffer Sim', ':6.3f')
        neig_similarity = AverageMeter('Buffer Neig Sim', ':6.3f')
        buff_meters = [num_seen, num_seen_max, similarity,
        neig_similarity]
    progress = ProgressMeter(len(trainloader), [batch_time, data_time, lr_meter] + buff_meters + [pos_pairs] + [losses] + [psc_losses] + [ird_losses],
                             prefix="Epoch: [{}]".format(epoch),
                             tbwriter=writer)


    # model を trainモード，model2 を evalモード に変更
    model.train()
    model2.eval()

    # criterions の分解
    criterion_psc = criterions["psc"]
    criterion_ird = criterions["ird"]


    # DDP 環境
    local_rank = cfg.ddp.local_rank
    device = torch.device(f"cuda:{local_rank}")

    # amp の使用状況
    use_amp = bool(getattr(cfg, "amp", None) and cfg.amp.use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and str(cfg.amp.dtype).lower() == "bf16") else torch.float16


    # 
    distill_power = 0.0


    print("len(trainloader): ", len(trainloader))

    end = time.time()
    for idx, data in enumerate(trainloader):

        if (idx % cfg.continual.repeat) == 0:
            model2 = copy.deepcopy(model)

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
            images_0 = images[0].to(device, non_blocking=True)
            images_1 = images[1].to(device, non_blocking=True)


        # forward 処理と損失計算
        if use_amp:
            with autocast(dtype=amp_dtype):
                
                # 蒸留損失の計算
                _, cur_features = model(images_0)
                _, past_features = model2(images_0)

                f0_logits, loss_ird = criterion_ird(cur_features, past_features, device)
                ird_losses.update(loss_ird.item(), images_0.size(0))

                
                # # 擬似教師あり対照損失の計算
                # all_images = torch.cat([images_0, images_1], dim=0)
                # # with torch.no_grad():
                # #     _, features_all = model(all_images)
                # _, features_all = model(all_images)
                # contrast_mask = similarity_mask_old(features_all, images_0.shape[0], cfg, pos_pairs, device)
                # # print("contrast_mask.shape: ", contrast_mask.shape)
                
                # _, features_0 = model(images_0)
                # _, features_1 = model(images_1)
                
                # loss_psc = criterion_psc(features_0, features_1, mask=contrast_mask, device=device)
                # psc_losses.update(loss_psc.item(), images_0.size(0))

                # # distill_powerの再計算
                # if distill_power <= 0 and loss_ird > 0.0:
                #     distill_power = psc_losses.avg * cfg.method.distill_power / ird_losses.avg

                # 損失を合計
                # loss = loss_psc + distill_power * loss_ird
                loss = distill_power * loss_ird
                losses.update(loss.item(), images_0.size(0))

                print("loss.shape: ", loss.shape)
                print("loss: ", loss)
                # assert False

        else:
            
            # 蒸留損失の計算
            _, cur_features = model(images_0)
            _, past_features = model2(images_0)

            f0_logits, loss_ird = criterion_ird(cur_features, past_features, device)
            ird_losses.update(loss_ird.item(), images_0.size(0))

            
            # 擬似教師あり対照損失の計算
            all_images = torch.cat([images_0, images_1], dim=0)
            # with torch.no_grad():
            #     _, features_all = model(all_images)
            _, features_all = model(all_images)
            contrast_mask = similarity_mask_old(features_all, images_0.shape[0], cfg, pos_pairs, device)
            # print("contrast_mask.shape: ", contrast_mask.shape)
            
            _, features_0 = model(images_0)
            _, features_1 = model(images_1)
            
            loss_psc = criterion_psc(features_0, features_1, mask=contrast_mask, device=device)
            psc_losses.update(loss_psc.item(), images_0.size(0))

            # distill_powerの再計算
            if distill_power <= 0 and loss_ird > 0.0:
                distill_power = psc_losses.avg * cfg.method.distill_power / ird_losses.avg
                
            # 損失を合計
            loss = loss_psc + distill_power * loss_ird
            losses.update(loss.item(), images_0.size(0))

            print("loss.shape: ", loss.shape)
            # assert False



        # ==============================
        # backward + step (AMP 対応)
        # ==============================
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


        # # バッファ内のサンプルの情報を更新する
        # with torch.no_grad():

        #     # 平均特徴をバッファ内のサンプル情報として使用するため
        #     sample_features = F.normalize(cur_features + features_2, p=2, dim=-1)
        #     data['feature'] = sample_features.detach()

        #     if cfg.continual.buffer_type in ["minred"]:
        #         stats = trainloader.batch_sampler.update_sample_stats(data)

        #         if 'num_seen' in stats:
        #             num_seen.update(stats['num_seen'].float().mean().item(),
        #                             stats['num_seen'].shape[0])
        #             num_seen_max.update(stats['num_seen'].float().max().item(),
        #                                 stats['num_seen'].shape[0])
        #         if 'similarity' in stats:
        #             similarity.update(stats['similarity'].float().mean().item(),
        #                               stats['similarity'].shape[0])
        #         if 'neighbor_similarity' in stats:
        #             neig_similarity.update(
        #                 stats['neighbor_similarity'].float().mean().item(),
        #                 stats['neighbor_similarity'].shape[0])
                    
        #     else:  
        #         assert False
        

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
        # print("len(trainloader.batch_sampler.buffer): ", len(trainloader.batch_sampler.buffer))
        if batch_i % cfg.log.print_freq == 0:
            tb_step = (epoch * len(trainloader.dataset) // cfg.optimizer.train.batch_size + batch_i * int(cfg.ddp.world_size))
            progress.display(batch_i)
            progress.tbwrite(tb_step)
