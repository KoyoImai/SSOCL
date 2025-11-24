

export CUDA_VISIBLE_DEVICES="0,1,2,3"


export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200

export CUDA_LAUNCH_BLOCKING=1


# python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --config-path ./configs/default/ --config-name default_ours
# python main_linear.py --config-path ./configs/default/ --config-name default_ours





# python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --config-path ./configs/default/ --config-name default_oursv2
python main_linear.py --config-path ./configs/default/ --config-name default_oursv2

