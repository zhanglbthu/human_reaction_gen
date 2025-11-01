export CUDA_VISIBLE_DEVICES=2

python train_mask_transformer.py \
    --name ar_dinoCond_traj_depth_1031 \
    --gpu_id 0 \
    --dataset_name vimo \
    --batch_size 64 \
    --max_epoch 50 \
    --vq_name rvq_official \
    --milestones 6000 \
    --warm_up_iter 250 \
    --n_layers 6 \
    --train_txt train_spatial.txt \
    --test_txt test_spatial.txt \
    --use_depth \
    --use_traj 