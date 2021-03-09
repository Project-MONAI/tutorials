# train step 1, with large learning rate

lr=1e-1
fold=0

python train.py -fold $fold -train_num_workers 4 -interval 5 -num_samples 4 \
-learning_rate $lr -max_epochs 1000 -task_id 05 -pos_sample_num 1 \
-expr_name baseline -tta_val True -determinism_flag True -determinism_seed 0