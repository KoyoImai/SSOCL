


import torch.nn as nn


def make_criterion(cfg):

    if cfg.method.name in ["ours"]:

        from losses.loss_ours import MultiCropContrastiveLoss, TotalCodingRateLoss

        mcc_criterion = MultiCropContrastiveLoss(temp=cfg.method.temp_mcc)
        tcr_criterion = TotalCodingRateLoss(eps=cfg.method.eps_tcr)

        criterions = {"mcc": mcc_criterion, "tcr": tcr_criterion}


    elif cfg.method.name in ["minred"]:
        criterion = nn.CosineSimilarity(dim=1)

        criterions = {"cos": criterion}
    
    elif cfg.method.name in ["empssl"]:

        from losses.loss_empssl import SimilarityLoss, TotalCodingRateLoss
        
        sim_criterion = SimilarityLoss()
        tcr_criterion = TotalCodingRateLoss(eps=cfg.method.eps_tcr)

        criterions = {"sim": sim_criterion, "tcr": tcr_criterion}


    
    else:
        assert False
    


    return criterions