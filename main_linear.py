
import os
import hydra


from models import make_model
from utils import seed_everything, CheckpointManager




def preparation(cfg):

    # データセット毎にタスク数・タスク毎のクラス数を決定
    # （現状不要）

    # 総タスク数
    # （現状不要）

    # モデルの保存，実験記録などの保存先パス
    if cfg.dataset.data_folder is None:
        cfg.dataset.data_folder = '~/data/'
    cfg.log.model_path = f'./logs/{cfg.method.name}/{cfg.log.name}/model/'      # modelの保存先
    cfg.log.explog_path = f'./logs/{cfg.method.name}/{cfg.log.name}/exp_log/'   # 実験記録の保存先
    cfg.log.mem_path = f'./logs/{cfg.method.name}/{cfg.log.name}/mem_log/'      # リプレイバッファ内の保存先
    cfg.log.result_path = f'./logs/{cfg.method.name}/{cfg.log.name}/result/'    # 結果の保存先

    # ディレクトリ作成
    if cfg.ddp.local_rank == 0:
        if not os.path.isdir(cfg.log.model_path):
            os.makedirs(cfg.log.model_path)
        if not os.path.isdir(cfg.log.explog_path):
            os.makedirs(cfg.log.explog_path)
        if not os.path.isdir(cfg.log.mem_path):
            os.makedirs(cfg.log.mem_path)
        if not os.path.isdir(cfg.log.result_path):
            os.makedirs(cfg.log.result_path)





@hydra.main(config_path='configs/default/', config_name='default', version_base=None)
def main(cfg):

    # ===========================================
    # シード値固定
    # ===========================================
    seed_everything(cfg.seed)


    # ===========================================
    # データローダ作成やディレクトリ作成などの前処理
    # ===========================================
    preparation(cfg)


    # ===========================================
    # DDP 関連の処理を実行
    # （一旦なし．可能な限り1gpuで実行する）
    # ===========================================


    # ===========================================
    # modelの作成
    # ===========================================
    model, _ = make_model(cfg)


    # ===========================================
    # 学習済みパラメータの読み込み
    # ===========================================
    







if __name__ == '__main__':
    main()



