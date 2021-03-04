# train step 2, finetune with small learning rate
# please replace the weight variable into your actual weight

lr=1e-3
fold=0
weight=model.pt

python train.py -fold $fold -train_num_workers 4 -interval 1 -num_samples 1 \
-learning_rate $lr -max_epochs 50 -task_id 04 -pos_sample_num 1 \
-expr_name baseline -tta_val True -checkpoint $weight -determinism_flag True \
-determinism_seed 0