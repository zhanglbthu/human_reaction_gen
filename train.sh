export CUDA_VISIBLE_DEVICES=0

python train_mask_transformer_dino.py \
    --name img_new \
    --gpu_id 0 \
    --dataset_name vimo \
    --batch_size 64 \
    --max_epoch 200 \
    --vq_name rvq_ourstrain \
    --milestones 6000 \
    --warm_up_iter 250 \
    --n_layers 6 \
    --train_txt train_spatial.txt \
    --test_txt test_spatial.txt \
    --dino_encoder vits \
    # --use_traj \
    # --use_depth