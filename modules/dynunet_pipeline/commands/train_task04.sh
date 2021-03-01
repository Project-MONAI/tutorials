lr=1e-1
# train step 1, with large learning rate
CUDA_VISIBLE_DEVICES=0 python train.py -fold 0 -train_num_workers 4 \
-interval 1 -num_samples 1 -learning_rate $lr -max_epochs 500 -task_id 04 \
-pos_sample_num 2 -expr_name baseline -tta_val True