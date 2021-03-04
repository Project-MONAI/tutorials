# please replace the weight variable into your actual weight

weight=model.pt
fold=0

python inference.py -fold $fold -expr_name baseline -task_id 04 -tta_val True \
-checkpoint $weight
