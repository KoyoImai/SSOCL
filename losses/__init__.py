

from losses.loss_ours import MultiCropContrastiveLoss, TotalCodingRateLoss




def make_criterion(cfg):

    if cfg.method.name in ["ours"]:

        mcc_criterion = MultiCropContrastiveLoss(temp=cfg.method.temp_mcc)
        tcr_criterion = TotalCodingRateLoss(eps=cfg.method.eps_tcr)

        criterions = {"mcc": mcc_criterion, "tcr": tcr_criterion}


    else:
        assert False
    




    return criterions