# train step 2, finetune with small learning rate
# please replace the weight variable into your actual weight

lr=1e-2
fold=0
weight=model.pt

python train.py -fold $fold -train_num_workers 4 -interval 5 -num_samples 3 \
-learning_rate $lr -max_epochs 1000 -task_id 09 -pos_sample_num 2 \
-expr_name baseline -tta_val True -checkpoint $weight -determinism_flag True \
-determinism_seed 0 -lr_decay True -batch_dice True
