# please replace the weight variable into your actual weight

weight=model.pt
fold=0

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_addr="localhost" --master_port=1234 \
	inference.py -fold $fold -expr_name baseline -task_id 04 -tta_val True \
	-checkpoint $weight -multi_gpu True
