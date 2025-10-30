export CUDA_VISIBLE_DEVICES=0

python train_mask_transformer.py \
    --name ar_firstImgConds \
    --gpu_id 0 \
    --dataset_name vimo \
    --batch_size 64 \
    --max_epoch 50 \
    --vq_name rvq_official \
    --milestones 6000 \
    --warm_up_iter 250 \
    --n_layers 6 \
    --train_txt train_spatial.txt \
    --test_txt test_spatial.txt