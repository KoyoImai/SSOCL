


import torch.nn as nn


from losses.loss_ours import MultiCropContrastiveLoss, TotalCodingRateLoss
from losses.loss_empssl import SimilarityLoss



def make_criterion(cfg):

    if cfg.method.name in ["ours"]:

        mcc_criterion = MultiCropContrastiveLoss(temp=cfg.method.temp_mcc)
        tcr_criterion = TotalCodingRateLoss(eps=cfg.method.eps_tcr)

        criterions = {"mcc": mcc_criterion, "tcr": tcr_criterion}


    elif cfg.method.name in ["minred"]:
        criterion = nn.CosineSimilarity(dim=1)

        criterions = {"cos": criterion}
    
    elif cfg.method.name in ["empssl"]:
        
        sim_criterion = SimilarityLoss()
        tcr_criterion = TotalCodingRateLoss(eps=cfg.method.eps_tcr)


    
    else:
        assert False
    

    


    return criterions