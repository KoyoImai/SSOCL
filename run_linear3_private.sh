
export CUDA_VISIBLE_DEVICES="3"


export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200

export CUDA_LAUNCH_BLOCKING=1


## ours4
# python main_linear.py --config-path ./configs/default/ --config-name default_ours4

# ## MinRed
python main_linear.py --config-path ./configs/default/ --config-name kcam_default_minred

# # empssl
# python main_linear.py --config-path ./configs/default/ --config-name default_empssl2

# # # minred4
# python main_linear.py --config-path ./configs/default/ --config-name default_minred4


# # empssl6
# python main_linear.py --config-path ./configs/default/ --config-name default_empssl6
