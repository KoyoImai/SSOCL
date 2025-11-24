
import os
import sys
import csv
import time
from tqdm import tqdm
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import torch
import torch.nn.functional as F

from utils import AverageMeter, ProgressMeter



def write_csv(value, path, file_name, task, ckpt=None):

    # ファイルパスを生成
    file_path = f"{path}/{file_name}_task{task}.csv"

    # ファイルが存在しなければ新規作成、かつヘッダー行を記入する
    # value がリストの場合は、ヘッダーの値部分は要素数に合わせて "value_1", "value_2", ... とする例
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # ヘッダー行を定義（必要に応じて適宜変更）
            if isinstance(value, list):
                header = ["ckpt"], ["task"] + [f"task_{i+1}" for i in range(len(value))]
            else:
                header = ["ckpt", "task", "value"]
            writer.writerow(header)

    # CSV に実際のデータを追加記録する
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if isinstance(value, list):
            row = [ckpt] + [task] + value
        else:
            row = [ckpt, task, value]
        writer.writerow(row)


def feature_extractor(model, trainloader, valloader, writer, cfg):

    # model を eval モードに変更
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # ----------------------------------------
    # 1. train データの feature bank 構築
    # ----------------------------------------
    # 訓練用のプログレスメーター
    batch_time = AverageMeter('Time', ':6.3f', tbname='train/time')
    data_time = AverageMeter('Data', ':6.3f', tbname='train/datatime')
    progress = ProgressMeter(len(trainloader),
                             [batch_time, data_time],
                             prefix="Train: ",
                             tbwriter=writer)
    
    train_features = []
    train_labels = []
    
    end = time.time()
    for idx, data in tqdm(enumerate(trainloader)):

        images = data["input"]
        labels = data["target"]

        # data loading time（あんまり使ってないけど一応）
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            feats = model.encoder(images)
            feats = F.normalize(feats, dim=1)
        

        # CPU 側に溜めておく（メモリ節約したければここはお好みで）
        train_features.append(feats.cpu())
        train_labels.append(labels.cpu())

        # 時間計測
        batch_time.update(time.time() - end)
        end = time.time()

        # if idx % cfg.log.print_freq == 0:
        #     progress.display(idx)
    
    # list -> Tensor にまとめる
    train_features = torch.cat(train_features, dim=0)   # [N_train, D]
    train_labels = torch.cat(train_labels, dim=0)       # [N_train]


    # ----------------------------------------
    # 2. val/test データの feature 抽出
    # ----------------------------------------
    batch_time = AverageMeter('Time', ':6.3f', tbname='val/time')
    data_time = AverageMeter('Data', ':6.3f', tbname='val/datatime')
    progress = ProgressMeter(
        len(valloader),
        [batch_time, data_time],
        prefix="Val(feat): ",
        tbwriter=writer
    )

    val_features = []
    val_labels = []

    end = time.time()
    for idx, data in tqdm(enumerate(valloader)):

        images = data["input"]
        labels = data["target"]

        if torch.cuda.is_available():
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            feats = model.encoder(images)
            feats = F.normalize(feats, dim=1)

        val_features.append(feats.cpu())
        val_labels.append(labels.cpu())
    
        batch_time.update(time.time() - end)
        end = time.time()

        # if idx % cfg.log.print_freq == 0:
        #     progress.display(idx)

    val_features = torch.cat(val_features, dim=0)   # [N_val, D]
    val_labels = torch.cat(val_labels, dim=0)       # [N_val]

    return train_features, train_labels, val_features, val_labels


def feature_label_save(train_features,
                       train_labels,
                       val_features,
                       val_labels,
                       cfg,
                       save_dir=None,
                       filename=None,
                     ):

    # ---------- 2. 保存先ディレクトリを決定 ----------
    if save_dir is None:
        # main.py / main_linear.py と同じ log の構造を使う
        # ./logs/<method.name>/<log.name>/result/knn_features/
        base_result = getattr(cfg.log, "result_path", "./logs")
        save_dir = os.path.join(base_result, "knn_features")

    os.makedirs(save_dir, exist_ok=True)

    # ---------- 3. ファイル名を決定 ----------
    if filename is None:
        ckpt = getattr(getattr(cfg, "linear", {}), "ckpt", None)
        if ckpt is not None:
            ckpt_stem = os.path.splitext(os.path.basename(ckpt))[0]
            filename = f"features_{ckpt_stem}.pt"
        else:
            filename = "features.pt"

    save_path = os.path.join(save_dir, filename)

    # ---------- 4. CPU に移して保存 ----------
    save_dict = {
        "train_features": train_features.cpu(),
        "train_labels": train_labels.cpu(),
        "val_features": val_features.cpu(),
        "val_labels": val_labels.cpu(),
    }

    torch.save(save_dict, save_path)
    print(f"[feature_label_save] Saved features to: {save_path}")

    return save_path



def knn_orig(cfg,
             train_features,
             train_labels,
             val_features,
             val_labels,
             k=20,
             T=0.07,
             chunk_size=1024):
    """
    k-NN による分類評価を行う関数。

    Args:
        train_features: Tensor, shape [N_train, D]
        train_labels:   LongTensor, shape [N_train]
        val_features:   Tensor, shape [N_val, D]
        val_labels:     LongTensor, shape [N_val]
        k:              近傍数 (k-NN の k)
        T:              温度 (similarity / T の softmax 的な重みづけに使用)
        chunk_size:     メモリ節約のため、val side をこのバッチサイズで分割して処理

    Returns:
        top1: float, top-1 accuracy (%)
        top5: float, top-5 accuracy (%)
    """
    ckpt = cfg.knn.ckpt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 念のため normalize（feature_extractor 側でも normalize 済みならここは冗長だけど安全）
    train_features = F.normalize(train_features, dim=1)
    val_features   = F.normalize(val_features, dim=1)

    # device に乗せる
    train_features = train_features.to(device)
    train_labels   = train_labels.to(device)

    # ラベルは 0 ~ C-1 の連番になっている前提
    num_classes = int(train_labels.max().item()) + 1

    total = 0
    top1_correct = 0
    top5_correct = 0

    # val_features を chunk に分割して処理
    num_val = val_features.size(0)
    for start in range(0, num_val, chunk_size):
        end = min(start + chunk_size, num_val)

        feats = val_features[start:end].to(device)          # [B, D]
        targets = val_labels[start:end].to(device)          # [B]

        # cosine similarity (feature はすでに normalize 済みなので内積が cosine)
        # sim: [B, N_train]
        sim = torch.mm(feats, train_features.t())

        # 上位 k 個の近傍を取得
        sim_weights, sim_indices = sim.topk(k=k, dim=1, largest=True, sorted=True)  # [B, k]

        # 近傍のラベルを取得
        neighbors = train_labels[sim_indices]               # [B, k]

        # 類似度に temperature をかけて softmax 的な重みづけ
        # (MoCo などでよく使われるやり方)
        sim_weights = (sim_weights / T).exp()               # [B, k]

        # クラスごとのスコア（weighted vote）
        # probs[b, c] = Σ_j w_{b,j} * 1[label_{b,j} == c]
        B = feats.size(0)
        probs = torch.zeros(B, num_classes, device=device)  # [B, C]
        probs.scatter_add_(dim=1, index=neighbors, src=sim_weights)

        # top-5 までの予測クラスを取得
        _, pred = probs.topk(k=5, dim=1, largest=True, sorted=True)  # [B, 5]

        # top-1
        top1_correct += (pred[:, 0] == targets).sum().item()
        # top-5: どれか一つでも正解なら OK
        top5_correct += (pred == targets.unsqueeze(1)).any(dim=1).sum().item()

        total += targets.size(0)

    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total

    if cfg.knn.task_id == []:
        write_csv(top1, cfg.log.result_path, file_name="top1_knn_acc", task="all", ckpt=ckpt)
        write_csv(top5, cfg.log.result_path, file_name="top5_knn_acc", task="all", ckpt=ckpt)
    else:
        write_csv(top1, cfg.log.result_path, file_name="top1_knn_acc", task=cfg.knn.task_id, ckpt=ckpt)
        write_csv(top5, cfg.log.result_path, file_name="top5_knn_acc", task=cfg.knn.task_id, ckpt=ckpt)

    return top1, top5




def knn(cfg,
        train_features,
        train_labels,
        val_features,
        val_labels,
        n_neighbors=20,
        metric="euclidean",   # feature を L2 normalize すれば euclidean で cosine と等価
        n_jobs=-1,
):
    ckpt = cfg.knn.ckpt

    # ---- 1. Tensor -> NumPy + normalize ----
    # ここで L2 normalize しておくと、euclidean 距離と cosine 類似度が単調変換で等価になる
    if isinstance(train_features, torch.Tensor):
        train_features = F.normalize(train_features, dim=1).cpu().numpy().astype(np.float32)
    if isinstance(val_features, torch.Tensor):
        val_features = F.normalize(val_features, dim=1).cpu().numpy().astype(np.float32)
    

    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu().numpy()

    # ---- 2. KNeighborsClassifier の学習 ----
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        n_jobs=n_jobs,
        algorithm="brute" if metric == "cosine" else "auto",  # cosine 使うなら brute のほうが安全
    )

    knn.fit(train_features, train_labels)

    # ---- 3. Top-1 accuracy ----
    val_pred = knn.predict(val_features)
    top1 = accuracy_score(val_labels, val_pred) * 100.0

    # ---- 4. 「Top-5っぽいもの」を計算（近傍ラベル中に正解が含まれるか）----
    k_for_top5 = min(5, n_neighbors)
    # k_neighbors をそのまま使うと、fit 時に使った neighbor 数 (n_neighbors) まで取れる
    distances, indices = knn.kneighbors(val_features, n_neighbors=k_for_top5)
    neighbor_labels = train_labels[indices]  # shape: [N_val, k_for_top5]

    # 各サンプルで「近傍のどれかのラベルが正解と一致しているか」
    correct_top5 = (neighbor_labels == val_labels[:, None]).any(axis=1)
    top5 = correct_top5.mean() * 100.0

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                top5=top5))

    if cfg.linear.task_id == []:
        write_csv(top1.avg.item(), cfg.log.result_path, file_name="top1_acc", task="all", ckpt=ckpt)
        write_csv(top5.avg.item(), cfg.log.result_path, file_name="top5_acc", task="all", ckpt=ckpt)
    else:
        write_csv(top1.avg.item(), cfg.log.result_path, file_name="top1_acc", task=cfg.knn.task_id, eckpt=ckpt)
        write_csv(top5.avg.item(), cfg.log.result_path, file_name="top5_acc", task=cfg.knn.task_id, ckpt=ckpt)


    return top1, top5



