


from train.train_ours import train_ours
from train.train_minred import train_minred



def train(model, model2, criterions, optimizer, trainloader, cfg, epoch, ckpt_manager, writer, scaler):

    if cfg.method.name == "ours":

        train_ours(model=model, model2=model2, criterions=criterions, optimizer=optimizer,
                   trainloader=trainloader, cfg=cfg, epoch=epoch, ckpt_manager=ckpt_manager, writer=writer, scaler=scaler)


    elif cfg.method.name == "minred":

        train_minred(model=model, model2=model2, criterions=criterions, optimizer=optimizer,
                     trainloader=trainloader, cfg=cfg, epoch=epoch, ckpt_manager=ckpt_manager, writer=writer, scaler=scaler)









