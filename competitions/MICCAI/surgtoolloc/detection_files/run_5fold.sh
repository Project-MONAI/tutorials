export CUDA_VISIBLE_DEVICES=0,1,2,3
img_size=512
workers=2
epochs=300
model="v5m"
for fold in 0 1 2 3 4
do
    python -m torch.distributed.run --nproc_per_node 4 train.py \
                                --img ${img_size} --batch 128 --epochs ${epochs} \
                                --data /raid/label_14_tools_yolo_640_blur/surg_14cls_fold${fold}.yaml \
                                --project ./surg_14cls_yolo${model}_5fold_outputs \
                                --name surg_${model}_${img_size}_fold${fold}${epochs}_epoch \
                                --weights yolo${model}.pt \
                                --device 0,1,2,3 \
                                --workers ${workers} \
                                --hyp ${model}_surg.yaml \
                                --sync-bn \
                                --optimizer 'Adam'
done
