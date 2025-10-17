


from dataloaders.imagenet21k import ImageNet21K
from dataloaders.stream_sampler import StreamSampler
from dataloaders.base_buffer_batchsampler import BaseBufferBatchSampler



def make_dataset(cfg, transform):

    if cfg.dataset.type == "imagenet21k":
        
        # def __init__(self, cfg, transforms=None, filelist=None, num_task=None, train=True):

        train_dataset = ImageNet21K(cfg=cfg,
                                    transform=transform,
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

    batchsampler = BaseBufferBatchSampler(buffer_size=cfg.continual.buffer_size, repeat=cfg.continual.repeat,
                                          dataset=dataset, sampler=sampler, batch_size=cfg.optimizer.train.batch_size)
    assert False





    # def __init__(self,
    #              buffer_size: int,
    #              repeat: int,
    #              dataset,
    #              sampler: Sampler[int],
    #              batch_size: int) -> None:





