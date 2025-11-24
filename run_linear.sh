

export CUDA_VISIBLE_DEVICES="3"


export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200

export CUDA_LAUNCH_BLOCKING=1


## ours4
python main_linear.py --config-path ./configs/default/ --config-name kcam_default_ours

# ## MinRed
# python main_linear.py --config-path ./configs/default/ --config-name default_minred2


# ## default_minred4
# python main_linear.py --config-path ./configs/default/ --config-name default_minred4 

# default_empssl6
# python main_linear.py --config-path ./configs/default/ --config-name default_empssl6

# ## ours5
# python main_linear.py --config-path ./configs/default/ --config-name default_ours5

# # default_empssl2
# python main_linear.py --config-path ./configs/default/ --config-name default_empssl2


