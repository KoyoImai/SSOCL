


from train.train_ours import train_ours



def train(model, model2, criterions, optimizer, trainloader, cfg):

    if cfg.method.name == "ours":

        train_ours(model=model, model2=model2, criterions=criterions, optimizer=optimizer, trainloader=trainloader, cfg=cfg)


    assert False











