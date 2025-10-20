


from train.train_ours import train_ours



def train(model, model2, criterions, optimizer, trainloader, cfg, epoch, ckpt_manager, writer):

    if cfg.method.name == "ours":

        train_ours(model=model, model2=model2, criterions=criterions, optimizer=optimizer,
                   trainloader=trainloader, cfg=cfg, epoch=epoch, ckpt_manager=ckpt_manager, writer=writer)


    assert False











