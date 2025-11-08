export CUDA_VISIBLE_DEVICES=1

python train_mask_transformer_memo_cross_vimo.py \
    --name baseline_1109 \
    --gpu_id 0 \
    --dataset_name vimo \
    --batch_size 64 \
    --max_epoch 15 \
    --vq_name rvq_official \
    --milestones 6000 \
    --warm_up_iter 250 \
    --n_layers 6 \
    --train_txt train_spatial.txt \
    --test_txt test_spatial.txt