

import torch

from utils import AverageMeter



def train_ours(model, model2, criterions, optimizer, trainloader, cfg):

    # model を trainモード，model2 を evalモード に変更
    model.train()
    model2.eval()

    # 学習記録
    losses = AverageMeter()
    accuracies = AverageMeter()

    # DDP 環境
    local_rank = cfg.ddp.local_rank
    device = torch.device(f"cuda:{local_rank}")



    for idx, data in enumerate(trainloader):
        
        images = data["input"]
        meta = data["meta"]
        labels = meta["labels"]

        images = torch.cat(images, dim=0)

        if torch.cuda.is_available():
            images = images.to(device, non_blocking=True)
        

        # model の forward 処理
        encoded, feature, z = model(images)
        if local_rank == 0:
            print("encoded.shape: ", encoded.shape)
            print("feature.shape: ", feature.shape)
            print("z.shape: ", z.shape)

        assert False


    assert False







