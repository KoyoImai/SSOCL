

export CUDA_VISIBLE_DEVICES="3"


export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200

export CUDA_LAUNCH_BLOCKING=1


## ours5
python main_knn.py --config-path ./configs/default/ --config-name default_ours5

