export CUDA_VISIBLE_DEVICES=2

python eval_trans_res_memo_cross_vimo.py \
    --dataset_name vimo \
    --vq_name rvq_official \
    --name ar_traj_depth_1108 \
    --exp_name ours_football \
    --res_name rtrans_official \
    --gpu_id 0 \
    --cond_scale 4 \
    --time_steps 10 \
    --ext rvq1_rtrans1_bs64_cs4_ts10 \
    --which_epoch all \
    --test_txt football.txt \
    # --use_depth 