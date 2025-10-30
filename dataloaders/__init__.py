


from dataloaders.imagenet21k import ImageNet21K, ImageNet21K_linear
from dataloaders.stream_sampler import StreamSampler
from dataloaders.batchsampler.base_buffer_batchsampler import BaseBufferBatchSampler
from dataloaders.batchsampler.minred_buffer_batchsampler import MinRedBufferBatchSampler




def make_dataset(cfg, transform):

    if cfg.dataset.type == "imagenet21k":
        
        # def __init__(self, cfg, transforms=None, filelist=None, num_task=None, train=True):

        train_dataset = ImageNet21K(cfg=cfg,
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

    if cfg.continual.buffer_type == "minred":

        batchsampler = MinRedBufferBatchSampler(buffer_size=cfg.continual.buffer_size, repeat=cfg.continual.repeat,
                                                dataset=dataset, sampler=sampler, batch_size=cfg.optimizer.train.batch_size, rank=cfg.ddp.local_rank)

    
    return batchsampler








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
