import sys
import os
from os.path import join as pjoin

import torch
from models.vq.model import RVQVAE
from options.vq_option import arg_parse
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
import utils.eval_vimo as eval_vimo
from utils.get_opt import get_opt
from models.t2m_eval_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from utils.word_vectorizer import WordVectorizer

def load_vq_model(vq_opt, which_epoch):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')

    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.code_dim,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', which_epoch),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
    print(f'Loading VQ Model {vq_opt.name} completed!, Epoch {vq_epoch}')
    return vq_model, vq_epoch

if __name__ == "__main__":
    ##### ---- Exp dirs ---- #####
    args = arg_parse(False)
    args.device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))

    args.out_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'eval')
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset_name == "vimo":
        args.nb_joints = 22
        dim_pose = 263
        dataset_opt_path = 'checkpoints/vimo/Comp_v6_KLD005/opt.txt'
    else:
        raise KeyError('Dataset Does Not Exist')

    f = open(pjoin(args.out_dir, '%s.log'%args.ext), 'w')

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'motions_test', device=args.device)

    print('len(eval_val_loader):', len(eval_val_loader))

    ##### ---- Network ---- #####
    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=args.device)
    # net = load_vq_model()

    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    for file in os.listdir(model_dir):
        # if not file.endswith('tar'):
        #     continue
        # if not file.startswith('net_best_fid'):
        #     continue
        if args.which_epoch != "all" and args.which_epoch not in file:
            continue
        print('loading checkpoint {}'.format(file))
        net, ep = load_vq_model(vq_opt, file)

        net.eval()
        net.cuda()

        fid = []
        div_real = []
        div = []
        mae = []

        repeat_time = 20
        for i in range(repeat_time):
            eval_fid, eval_div_real, eval_div, dist = \
                eval_vimo.evaluation_vqvae_plus_mpjpe_vimo(eval_val_loader, net, i, eval_wrapper=eval_wrapper, num_joints=args.nb_joints)

            fid.append(eval_fid)
            div_real.append(eval_div_real)
            div.append(eval_div)
            mae.append(dist)

        fid = np.array(fid)
        div_real = np.array(div_real)
        div = np.array(div)
        mae = np.array(mae)

        print(f'{file} final result, epoch {ep}')
        print(f'{file} final result, epoch {ep}', file=f, flush=True)

        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity Real: {np.mean(div_real):.3f}, conf. {np.std(div_real)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMPJPE:{np.mean(mae):.4f}, conf.{np.std(mae)*1.96/np.sqrt(repeat_time):.4f}\n\n"
        # logger.info(msg_final)
        print(msg_final)
        print(msg_final, file=f, flush=True)

    f.close()

'''
python eval_vq_vimo.py --gpu_id 0 --name rvq_bs256_finetune_ep10 --dataset_name vimo --ext rvq_nq6
'''