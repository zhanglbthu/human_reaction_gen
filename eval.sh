export CUDA_VISIBLE_DEVICES=0

python eval_trans_res_memo_cross_vimo.py \
    --dataset_name vimo \
    --vq_name rvq_bs256_finetune_ep10 \
    --name mtrans_baseline \
    --res_name rtrans_memo_cross_l6_bs64_ep200 \
    --exp_name baseline_rvq_spatialExplosion \
    --gpu_id 0 \
    --cond_scale 4 \
    --time_steps 10 \
    --ext rvq1_rtrans1_bs64_cs4_ts10 \
    --which_epoch all \
    --test_txt test_spatial_explosion.txt