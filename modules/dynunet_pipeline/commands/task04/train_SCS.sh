# train step 1, with large learning rate

lr=1e-1
fold=0
train_num_workers=4
max_epochs=10
root_dir=/data/psma-pet/code/data/

python train.py -fold $fold -train_num_workers $train_num_workers -interval 1 -num_samples 1 \
-learning_rate $lr -max_epochs $max_epochs -task_id 04 -pos_sample_num 2 \
-expr_name baseline -tta_val True -determinism_flag True -determinism_seed 0 \
-root_dir $root_dir