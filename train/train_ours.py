

import torch
import torch.nn.functional as F


from utils import AverageMeter



def train_ours(model, model2, criterions, optimizer, trainloader, cfg):

    # model を trainモード，model2 を evalモード に変更
    model.train()
    model2.eval()

    # criterions の分解
    criterion_mcc = criterions["mcc"]
    criterion_tcr = criterions["tcr"]

    # 学習記録
    losses = AverageMeter()
    accuracies = AverageMeter()

    # DDP 環境
    local_rank = cfg.ddp.local_rank
    device = torch.device(f"cuda:{local_rank}")


    print("len(trainloader): ", len(trainloader))
    for idx, data in enumerate(trainloader):

        # 学習済みのバッチ数をカウント
        batch_i = trainloader.batch_sampler.advance_batches_seen()
        # init_batch_i = trainloader.batch_sampler.init_batch_i
        # batch_i = batch_i + init_batch_

        # 現在のタスクidを確認
        task_id = trainloader.batch_sampler.task_id
        print("task_id: ", task_id)


        # 画像とラベルを獲得
        images = data["input"]
        meta = data["meta"]
        labels = meta["label"]

        # マルチクロップ画像の連結
        images = torch.cat(images, dim=0)

        # gpuに配置
        if torch.cuda.is_available():
            images = images.to(device, non_blocking=True)
        

        # model の forward 処理
        encoded, feature, z_proj = model(images)
        # if local_rank == 0:
        #     print("encoded.shape: ", encoded.shape)
        #     print("feature.shape: ", feature.shape)
        #     print("z_proj.shape: ", z_proj.shape)

        # 特徴量平均計算
        z_list = z_proj.chunk(cfg.method.num_crops, dim=0)
        z_avg = chunk_avg(z_proj, cfg.method.num_crops)

        # MCC損失とTCR損失の計算
        loss_mcc = criterion_mcc(z_list)
        loss_tcr = cal_TCR(z_proj, criterion_tcr, cfg.method.num_crops)

        # 損失の合算
        loss = cfg.method.lambda_mcc * loss_mcc + cfg.method.lambda_tcr * loss_tcr
        # if local_rank == 0:
        #     print("loss_mcc: ", loss_mcc)
        #     print("loss_tcr: ", loss_tcr)
        #     print("loss: ", loss)
        

        # 最適化ステップ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        # バッファ内のサンプルの情報を更新する
        with torch.no_grad():

            # 平均特徴をバッファ内のサンプル情報として使用するため
            data["feature"] = z_avg.detach()

            if cfg.continual.buffer_type in ["minred"]:
                stats = trainloader.batch_sampler.update_sample_stats(data)
            else:  
                assert False

        # 後から分析可能にするため，学習途中のモデルを一定間隔で保存する
        # （未実装）
        

        # 現在の学習状況を記録しておき，途中から再開可能にする．
        # （未実装）




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






