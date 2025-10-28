export CUDA_VISIBLE_DEVICES=0

python train_mask_transformer_dino.py \
    --name img_1026_oldgen \
    --gpu_id 0 \
    --dataset_name vimo \
    --batch_size 64 \
    --max_epoch 10 \
    --vq_name rvq_official \
    --milestones 6000 \
    --warm_up_iter 250 \
    --n_layers 6 \
    --train_txt train_debug.txt \
    --test_txt test_debug.txt \
    --dino_encoder vits \
    # --use_traj \
    # --use_depth