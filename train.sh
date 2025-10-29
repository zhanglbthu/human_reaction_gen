export CUDA_VISIBLE_DEVICES=1

python train_mask_transformer_memo_cross_vimo.py \
    --name baseline_wofix \
    --gpu_id 0 \
    --dataset_name vimo \
    --batch_size 64 \
    --max_epoch 15 \
    --vq_name rvq_official \
    --milestones 6000 \
    --warm_up_iter 250 \
    --n_layers 6 \
    --train_txt train.txt \
    --test_txt test.txt