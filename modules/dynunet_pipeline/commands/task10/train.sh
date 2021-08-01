# train step 1, with large learning rate

lr=1e-2
fold=0

python train.py -fold $fold -train_num_workers 2 -interval 10 -num_samples 1 \
-learning_rate $lr -max_epochs 5000 -task_id 10 -pos_sample_num 1 \
-expr_name baseline -tta_val True -determinism_flag True -determinism_seed 0 \
-lr_decay True