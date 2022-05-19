# train step 1, with large learning rate

lr=1e-2
fold=0

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_addr="localhost" --master_port=1234 \
	train.py -fold $fold -train_num_workers 4 -interval 10 -num_samples 1 \
	-learning_rate $lr -max_epochs 2000 -task_id 07 -pos_sample_num 1 \
	-expr_name baseline -tta_val True -multi_gpu True -lr_decay True
