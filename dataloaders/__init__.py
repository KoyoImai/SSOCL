
import os

from dataloaders.imagenet21k import ImageNet21K, ImageNet21K_linear
from dataloaders.krishnacam import KrishnaCAM
from dataloaders.coco_detection import CocoDetectionDataset

from dataloaders.stream_sampler import StreamSampler
from dataloaders.batchsampler.minred_buffer_batchsampler import MinRedBufferBatchSampler
from dataloaders.batchsampler.random_buffer_batchsampler import RandomBufferBatchSampler
from dataloaders.batchsampler.mix_minred_buffer_batchsampler import MixMinRedBufferBatchSampler



def make_dataset(cfg, transform):

    if cfg.dataset.type == "imagenet21k":
        
        # def __init__(self, cfg, transforms=None, filelist=None, num_task=None, train=True):

        train_dataset = ImageNet21K(cfg=cfg,
                                    transforms=transform,
                                    filelist=cfg.dataset.filelist,
                                    num_task=cfg.continual.n_task,
                                    train=True)
    
    elif cfg.dataset.type == "krishnacam":

        train_dataset = KrishnaCAM(cfg=cfg,
                                   transforms=transform,
                                   filelist=cfg.dataset.filelist,
                                   num_task=cfg.continual.n_task,
                                   train=True)
    else:
        
        assert False

    

    return train_dataset



def make_sampler(cfg, dataset):

    sampler = StreamSampler(cfg=cfg, dataset=dataset, rank=cfg.ddp.local_rank, world_size=cfg.ddp.world_size,
                            drop_last=cfg.dataset.drop_last, base_seed=cfg.seed, sharding=cfg.dataset.sharding, start_index=cfg.dataset.start_index)


    return sampler


def make_batchsampler(cfg, dataset, sampler):

    if cfg.method.name in ["ours", "minred", "empssl"]:

        if cfg.continual.buffer_type == "minred":

            batchsampler = MinRedBufferBatchSampler(buffer_size=cfg.continual.buffer_size, repeat=cfg.continual.repeat,
                                                    dataset=dataset, sampler=sampler, batch_size=cfg.optimizer.train.batch_size, rank=cfg.ddp.local_rank)

        elif cfg.continual.buffer_type == "random":

            batchsampler = RandomBufferBatchSampler(buffer_size=cfg.continual.buffer_size, repeat=cfg.continual.repeat,
                                                    dataset=dataset, sampler=sampler, batch_size=cfg.optimizer.train.batch_size, rank=cfg.ddp.local_rank)

    elif cfg.method.name in ["scale"]:

        if cfg.continual.buffer_type == "minred":

            batchsampler = MixMinRedBufferBatchSampler(buffer_size=cfg.continual.buffer_size, repeat=cfg.continual.repeat, dataset=dataset, sampler=sampler,
                                                       batch_size=cfg.optimizer.train.stream_batch_size, mem_batch_size=cfg.optimizer.train.mem_batch_size, rank=cfg.ddp.local_rank)

    else:
        assert False
    
    return batchsampler




# ===========================================
# ここから下は評価用データセットの作成
# ===========================================

# 線形分類用のデータセット作成
def make_dataset_eval(cfg, train_transform, val_transform):

    if cfg.linear.task_id == []:
        task_id = list(range(cfg.linear.n_task))
    else:
        task_id = cfg.linear.task_id

    if cfg.dataset.type == "imagenet21k":

        train_dataset = ImageNet21K_linear(cfg=cfg,
                                           transforms=train_transform,
                                           filelist=cfg.dataset.filelist,
                                           num_task=cfg.continual.n_task,
                                           train=False,
                                           linear_train=True,
                                           task_id=task_id)
        
        val_dataset = ImageNet21K_linear(cfg=cfg,
                                         transforms=val_transform,
                                         filelist=cfg.dataset.filelist,
                                         num_task=cfg.continual.n_task,
                                         train=False,
                                         linear_train=False,
                                         task_id=task_id)
        

        # print("len(train_dataset): ", len(train_dataset))
        # print("len(val_dataset): ", len(val_dataset))

    else:
        assert False




    return train_dataset, val_dataset



# 物体検出用のデータセット作成
def make_detection_dataset(cfg, train_augmentation, test_augmentation):

    image_folder = cfg.detection.dataset.image_folder
    anno_folder = cfg.detection.dataset.anno_folder

    cfg.detection.dataset.train_folder = os.path.join(image_folder, "train2017")
    cfg.detection.dataset.test_folder = os.path.join(image_folder, "val2017")

    cfg.detection.dataset.train_ann = os.path.join(anno_folder, "instances_train2017.json")
    cfg.detection.dataset.test_ann = os.path.join(anno_folder, "instances_val2017.json")


    train_dataset = CocoDetectionDataset(
        image_dir=cfg.detection.dataset.train_folder,
        ann_file=cfg.detection.dataset.train_ann,
        transforms=train_augmentation,
    )
    test_dataset = CocoDetectionDataset(
        image_dir=cfg.detection.dataset.test_folder,
        ann_file=cfg.detection.dataset.test_ann,
        transforms=test_augmentation,
    )


    return train_dataset, test_dataset






