

import math


from train.train_ours import train_ours
from train.train_minred import train_minred
from train.train_empssl import train_empssl
from train.train_linear import train_linear
from train.eval_linear import eval_linear






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

        assert False




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



def linear_train(model, classifier, criterion, optimizer, trainloader, valloader, epoch, scaler, writer, cfg):

    train_linear(model=model, classifier=classifier, criterion=criterion, optimizer=optimizer,
                 trainloader=trainloader, valloader=valloader, epoch=epoch, scaler=scaler, writer=writer, cfg=cfg)


    return


def linear_eval(model, classifier, criterion, optimizer, trainloader, valloader, epoch, scaler, writer, cfg):

    eval_linear(model=model, classifier=classifier, criterion=criterion, optimizer=optimizer,
                 trainloader=trainloader, valloader=valloader, epoch=epoch, scaler=scaler, writer=writer, cfg=cfg)


    return 




