
import os
import time
from tqdm import tqdm
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert


# ★あなたの Normalize と同じ値に変更（例: ImageNet）
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def _denorm(x, mean=MEAN, std=STD):
    m = torch.tensor(mean, device=x.device).view(-1,1,1)
    s = torch.tensor(std,  device=x.device).view(-1,1,1)
    return (x * s + m).clamp(0, 1)  # CHW, 0-1へ




@torch.no_grad()
def visualize_batch(images, targets, outdir="debug_simple", n=4):
    """images: list[Tensor CHW], targets: list[dict] (torchvision detection 形式)"""
    os.makedirs(outdir, exist_ok=True)
    n = min(n, len(images))

    # 画像とBBoxを保存、簡単な統計を表示
    for i in range(n):
        x = images[i].detach().cpu()         # CHW, 正規化後
        t = targets[i]

        print(f"[{i}] shape={tuple(x.shape)} "
              f"min={x.min():.2f} max={x.max():.2f} mean={x.mean():.2f} std={x.std():.2f}")

        x01   = _denorm(x)                   # 正規化を戻す(0-1)
        img_u8= (x01*255).clamp(0,255).to(torch.uint8)

        boxes  = t.get("boxes", None)
        labels = t.get("labels", None)

        if boxes is not None and boxes.numel() > 0:
            texts = [str(int(l)) for l in labels] if labels is not None else None
            drawn = draw_bounding_boxes(img_u8, boxes.cpu(), labels=texts, width=2)
        else:
            drawn = img_u8

        TF.to_pil_image(drawn).save(os.path.join(outdir, f"sample_{i}.png"))

    # バッチ内のラベル分布（あれば）
    label_lists = [t["labels"].cpu() for t in targets if "labels" in t and len(t["labels"])]
    if label_lists:
        cnt = Counter(torch.cat(label_lists).tolist())
        print("Label counts:", dict(cnt))



def train_detection(model, train_loader, train_folder, optimizer, lr_scheduler, device, scaler, epoch, print_freq, cfg):

    # model を訓練モードに変更
    model.train()
    header = f"Epoch [{epoch}]"
    time_start = time.time()

    for i, (images, targets) in enumerate(train_loader):

        # if i > 200:
        #     break

        if lr_scheduler is not None:
            lr_scheduler.step()

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # デバッグ用にバッチの確認
        # visualize_batch(images, targets, outdir=f"debug_e{epoch:03d}_b{i:03d}", n=4)

        # print("images[0].shape: ", images[0].shape)    # images[0].shape:  torch.Size([3, 480, 640])
        # assert False

        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())


        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        # if scaler is not None:
        #     scaler.scale(losses).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        #     losses.backward()
        #     optimizer.step()

        if (i + 1) % print_freq == 0:
            elapsed = time.time() - time_start
            loss_str = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"{header} Iter {i + 1}/{len(train_loader)} | Loss {losses.item():.4f} | {loss_str} | {elapsed:.1f}s")





def eval_detection(model, test_loader, test_folder, optimizer, lr_scheduler, device, scaler, epoch, print_freq, cfg):

    model.eval()
    coco_gt: COCO = test_loader.dataset.dataset.coco
    results: List[Dict] = []

    count = 0
    with torch.no_grad():
        for images, targets in tqdm(test_loader):

            count += 1
            images = [img.to(device) for img in images]

            # # デバッグ用にバッチの確認
            # targets_vis = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # visualize_batch(images, targets_vis, outdir=f"debug_eval_e{epoch:03d}_b{count:03d}", n=4)

            outputs = model(images)

            for output, target in zip(outputs, targets):
                boxes = box_convert(output["boxes"].cpu(), in_fmt="xyxy", out_fmt="xywh")
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()
                image_id = int(target["image_id"].item())

                for box, score, label in zip(boxes, scores, labels):
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label.item()),
                            "bbox": box.tolist(),
                            "score": float(score.item()),
                        }
                    )

    if not results:
        return {"map": 0.0}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = {
        "map": coco_eval.stats[0],
        "map_50": coco_eval.stats[1],
        "map_75": coco_eval.stats[2],
        "map_small": coco_eval.stats[3],
        "map_medium": coco_eval.stats[4],
        "map_large": coco_eval.stats[5],
    }
    return stats




