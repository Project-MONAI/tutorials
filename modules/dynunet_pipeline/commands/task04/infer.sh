# please replace the weight variable into your actual weight

weight=model.pt
CUDA_VISIBLE_DEVICES=0 python inference.py -fold 0 -expr_name baseline -task_id 04 -tta_val True -checkpoint $weight
