# train step 2, finetune with small learning rate
# please replace the weight variable into your actual weight
# since this task uses lr scheduler, please set the lr and max epochs
# here according to the step 1 training results. The value of max epochs equals
# to 2000 minus the best epoch in step 1.

lr=5e-3
max_epochs=1000
fold=0
weight=model.pt

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_addr="localhost" --master_port=1234 \
    train.py -fold $fold -train_num_workers 4 -interval 10 -num_samples 1 \
    -learning_rate $lr -max_epochs $max_epochs -task_id 07 -pos_sample_num 1 \
    -expr_name baseline -tta_val True -checkpoint $weight -multi_gpu True \
    -lr_decay True
