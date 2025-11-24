
import os
import math

import torch

from train.train_ours import train_ours
from train.train_minred import train_minred
from train.train_empssl import train_empssl
from train.train_scale import train_scale

# 評価用
from train.train_linear import train_linear
from train.eval_linear import eval_linear
from train.knn import feature_extractor, feature_label_save, knn, knn_orig
from train.train_detection import train_detection, eval_detection





# ===========================================
# 事前学習
# ===========================================
def train(model, model2, criterions, optimizer, trainloader, cfg, epoch, ckpt_manager, writer, scaler):

    if cfg.method.name == "ours":

        train_ours(model=model, model2=model2, criterions=criterions, optimizer=optimizer,
                   trainloader=trainloader, cfg=cfg, epoch=epoch, ckpt_manager=ckpt_manager, writer=writer, scaler=scaler)


    elif cfg.method.name == "minred":

        train_minred(model=model, model2=model2, criterions=criterions, optimizer=optimizer,
                     trainloader=trainloader, cfg=cfg, epoch=epoch, ckpt_manager=ckpt_manager, writer=writer, scaler=scaler)


    elif cfg.method.name == "empssl":

        train_empssl(model=model, model2=model2, criterions=criterions, optimizer=optimizer,
                     trainloader=trainloader, cfg=cfg, epoch=epoch, ckpt_manager=ckpt_manager, writer=writer, scaler=scaler)

    elif cfg.method.name == "scale":

        train_scale(model=model, model2=model2, criterions=criterions, optimizer=optimizer,
                     trainloader=trainloader, cfg=cfg, epoch=epoch, ckpt_manager=ckpt_manager, writer=writer, scaler=scaler)




def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""
    lr = cfg.linear.train.learning_rate
    if cfg.linear.train.cos:
        lr *= 0.5 * (1. + math.cos(
            math.pi * epoch / cfg.linear.train.epochs))
    else:
        for milestone in cfg.linear.train.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




# ===========================================
# 線形分類訓練と評価
# ===========================================
def linear_train(model, classifier, criterion, optimizer, trainloader, valloader, epoch, scaler, writer, cfg):

    train_linear(model=model, classifier=classifier, criterion=criterion, optimizer=optimizer,
                 trainloader=trainloader, valloader=valloader, epoch=epoch, scaler=scaler, writer=writer, cfg=cfg)


    return


def linear_eval(model, classifier, criterion, optimizer, trainloader, valloader, epoch, scaler, writer, cfg):

    top1_acc = eval_linear(model=model, classifier=classifier, criterion=criterion, optimizer=optimizer,
                 trainloader=trainloader, valloader=valloader, epoch=epoch, scaler=scaler, writer=writer, cfg=cfg)


    return top1_acc



# ===========================================
# k-NN分類訓練と評価
# ===========================================
def knn_eval(model, trainloader, valloader, writer, cfg):

    # 訓練用データとテスト用データの特徴量・ラベルを保存するファイルのパス
    save_dir = cfg.log.explog_path

    filename = f"feature_label_taksid{cfg.knn.task_id}_{cfg.knn.ckpt}"
    file_path = os.path.join(save_dir, filename)

    # 特徴量がない場合，特徴抽出を実行
    if not os.path.isfile(file_path):
        train_features, train_labels, val_features, val_labels = feature_extractor(model, trainloader, valloader, writer, cfg)
        save_path = feature_label_save(train_features, train_labels, val_features, val_labels, cfg, save_dir=save_dir, filename=filename)
    # 特徴量が既に抽出済みならそれを読み込む
    else:
        data = torch.load(file_path, map_location="cpu")

        train_features = data["train_features"]
        train_labels   = data["train_labels"]
        val_features   = data["val_features"]
        val_labels     = data["val_labels"]
    print("train_features.shape: ", train_features.shape)
    print("train_labels.shape: ", train_labels.shape)
    print("val_features.shape: ", val_features.shape)
    print("val_labels.shape: ", val_labels.shape)


    # top1, top5 = knn(cfg,
    #                  train_features,
    #                  train_labels,
    #                  val_features,
    #                  val_labels,
    #                  n_neighbors=cfg.knn.num_k,
    #                  metric=cfg.knn.metric)
    top1, top5 = knn_orig(cfg,
                          train_features,
                          train_labels,
                          val_features,
                          val_labels,
                          k=cfg.knn.num_k,
                          T=0.07,
                          chunk_size=1024,
                         )


# ===========================================
# 物体検出の訓練と評価
# ===========================================
def detector_train(model, train_loader, train_folder, optimizer, lr_scheduler, device, scaler, epoch, print_freq, cfg):

    train_detection(model, train_loader, train_folder, optimizer, lr_scheduler, device, scaler, epoch, print_freq, cfg)
    



def detector_eval(model, test_loader, test_folder, optimizer, lr_scheduler, device, scaler, epoch, print_freq, cfg):

    metrics = eval_detection(model, test_loader, test_folder, optimizer, lr_scheduler, device, scaler, epoch, print_freq, cfg)

    return metrics




