

export CUDA_VISIBLE_DEVICES="3"


export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200



## default_empssl2
# python main_detection.py --config-path ./configs/default/ --config-name default_empssl2 > empssl_detection_debug.txt


## default_ours4
# python main_detection.py --config-path ./configs/default/ --config-name default_ours4 > ours4_detection.txt



# ## default_minred2
# python main_detection.py --config-path ./configs/default/ --config-name default_minred2 > minred2_detection.txt


## default_empssl2
# python main_detection.py --config-path ./configs/default/ --config-name default_empssl2 > empssl_detection_debug.txt


# ## default_minred4
# python main_detection.py --config-path ./configs/default/ --config-name default_minred4 > minred4_detection.txt



## default_ours5
python main_detection.py --config-path ./configs/default/ --config-name default_ours5 > ours5_detection_7200_layer3.txt


