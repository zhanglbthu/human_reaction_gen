export CUDA_VISIBLE_DEVICES=0

python gen.py \
    --dataset_name vimo \
    --vq_name rvq_bs256_finetune_ep10 \
    --name mtrans_memo_cross_l6_bs64_ep200 \
    --res_name rtrans_memo_cross_l6_bs64_ep200 \
    --ext gen_debug \
    --gpu_id 0 \
    --motion_length 100 \
    --video_path Data/VIMO/videos/chat/chat-001.mp4