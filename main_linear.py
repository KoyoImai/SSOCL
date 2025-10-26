

import hydra


from utils import seed_everything, CheckpointManager


@hydra.main(config_path='configs/default/', config_name='default', version_base=None)
def main(cfg):

    # ===========================================
    # シード値固定
    # ===========================================
    seed_everything(cfg.seed)






if __name__ == '__main__':
    main()



