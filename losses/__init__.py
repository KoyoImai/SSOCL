




def make_criterion(cfg):

    if cfg.method.name in ["ours"]:

        mcc_criterion = None
        tcr_criterion = None

        criterions = {"mcc": mcc_criterion, "tcr": tcr_criterion}


    else:
        assert False
    




    return criterions