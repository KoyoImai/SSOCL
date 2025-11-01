

export CUDA_VISIBLE_DEVICES="0"


export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200


# python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --config-path ./configs/default/ --config-name default_empssl2
# python main_linear.py --config-path ./configs/default/ --config-name default_empssl2
python main_detection.py --config-path ./configs/default/ --config-name default_empssl2 > empssl_detection_debug.txt
