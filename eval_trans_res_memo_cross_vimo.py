import os
from os.path import join as pjoin

import torch

from models.mask_transformer.transformer_memo_cross import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from data.vimo_dataset import VimoDataset, val_pipeline
from torch.utils.data import DataLoader
from models.tc_clip.tc_clip import TCCLIP_VE
from models.dinov2.dino_encoder import Dino_Encoder
from custom_clip import clip

import utils.eval_vimo as eval_vimo
from utils.fixseed import fixseed

import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils import paramUtil
kinematic_chain = paramUtil.t2m_kinematic_chain
fps = 20
radius = 4

def plot_t2m(data, save_dir):
    data = data * std + mean
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, kinematic_chain, joint, title="", fps=fps, radius=radius)

def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
    print(f'Loading VQ Model {vq_opt.name} from epoch {vq_epoch} completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, which_model):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='video',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=opt.device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    # assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]} completed!')
    return t2m_transformer, ckpt["ep"]

def load_res_model(res_opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='video',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_loss.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    # assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]} completed!')
    return res_transformer

def load_clip_to_cpu(model_arch):
    print(f'Loading CLIP Model with {model_arch}')
    backbone_name = model_arch
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, './checkpoints/clip')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"vision_model": "TCVisionTransformer",
                    "vision_block": "TCAttentionBlock",
                    "text_block": "ResidualAttentionBlock",
                    "use_custom_attention": True,
                    "context_length": 30,
                    "temporal_length": 50,
                    "vision_depth": 0,
                    "language_depth": 1,
                    "vision_ctx": 0,
                    "language_ctx": 0,
                    # TC-CLIP
                    "positional_embedding_type": "space",
                    "local_global_bias": True,
                    "context_token_k": 96,
                    "seed_token_a": 0.3,
                    "tome_r": 100,
                    "tome_d": 0
                    }

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def load_tcclip_model(model_path, model):
    print(f"Loading tcclip model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    load_state_dict = checkpoint['model']

    for param_name in list(load_state_dict.keys()):
        if 'image_encoder' not in param_name:
            del load_state_dict[param_name]
    
    msg = model.load_state_dict(load_state_dict, strict=False)
    print(f"loaded model: {msg}")
    del checkpoint
    torch.cuda.empty_cache()
    return model

def prepare_video_encoder(clip_version):
    clip_model = load_clip_to_cpu(clip_version)
    video_encoder = TCCLIP_VE(clip_model)

    for param in video_encoder.parameters():
        param.requires_grad_(False)
    
    video_encoder.float()
    del clip_model
    torch.cuda.empty_cache()
    video_encoder.to(opt.device)

    video_encoder = load_tcclip_model('./checkpoints/tcclip/zero_shot_k400_tc_clip_newkeys.pth', video_encoder)
    return video_encoder

def prepare_dino_encoder(encoder='vits'):
    # dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
    # dim = dino.blocks[0].attn.qkv.in_features
    # print(f'Loading DINOv2 {encoder} with feature dim {dim}')
    dino_encoder = Dino_Encoder(encoder=encoder)
    for param in dino_encoder.parameters():
        param.requires_grad_(False)
    dino_encoder.float()
    dino_encoder.to(opt.device)
    return dino_encoder

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    out_dir = pjoin(root_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    if opt.dataset_name == "vimo":
        opt.data_root = './Data/VIMO/'
        opt.nb_joints = 22
        dim_pose = 263
        dataset_opt_path = './checkpoints/vimo/Comp_v6_KLD005/opt.txt'
        clip_version = 'ViT-B/16'
    else:
        raise KeyError('Dataset Does Not Exist')

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    out_path = pjoin(out_dir, "%s.log"%opt.ext)

    f = open(pjoin(out_path), 'w')

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt)

    assert res_opt.vq_name == model_opt.vq_name

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # video_encoder = prepare_video_encoder(clip_version)
    video_encoder = prepare_dino_encoder(opt.dino_encoder)

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))

    eval_val_dataset = VimoDataset(opt, mean, std, data_prefix=opt.data_root, ann_file=opt.test_txt, pipeline=val_pipeline)
    eval_val_loader = DataLoader(eval_val_dataset, batch_size=opt.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    print('Preprocessing data...')
    eval_val_loader = [batch_data for batch_data in tqdm(eval_val_loader)]

    out_dir = './Data/eval'
    out_dir = os.path.join(out_dir, opt.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # model_dir = pjoin(opt.)
    for file in os.listdir(model_dir):
        if opt.which_epoch != "all" and opt.which_epoch not in file:
            continue
        if file != "net_best_fid.tar":
            continue
        print('loading checkpoint {}'.format(file))
        t2m_transformer, ep = load_trans_model(model_opt, file)

        t2m_transformer.eval()
        vq_model.eval()
        res_model.eval()
        video_encoder.eval()

        t2m_transformer.to(opt.device)
        vq_model.to(opt.device)
        res_model.to(opt.device)
        video_encoder.to(opt.device)

        fid = []
        div_real = []
        div = []
        mm = []

        repeat_time = 20
        for i in tqdm(range(repeat_time)):
            with torch.no_grad():
                eval_fid, eval_div_real, eval_div, eval_mm = \
                    eval_vimo.evaluation_mask_transformer_test_plus_res_memo(eval_val_loader, vq_model, res_model, t2m_transformer, video_encoder,
                                                                        i, eval_wrapper=eval_wrapper, time_steps=opt.time_steps,
                                                                        cond_scale=opt.cond_scale, temperature=opt.temperature, topkr=opt.topkr,
                                                                        gsample=opt.gumbel_sample, force_mask=opt.force_mask, 
                                                                        cal_mm=True,
                                                                        save_anim=False, 
                                                                        out_dir=out_dir, 
                                                                        plot_func=plot_t2m)
            fid.append(eval_fid)
            div_real.append(eval_div_real)
            div.append(eval_div)
            mm.append(eval_mm)

        fid = np.array(fid)
        div_real = np.array(div_real)
        div = np.array(div)
        mm = np.array(mm)

        print(f'{file} final result, epoch {ep}')
        print(f'{file} final result, epoch {ep}', file=f, flush=True)

        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity Real: {np.mean(div_real):.3f}, conf. {np.std(div_real)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
        # logger.info(msg_final)
        print(msg_final)
        print(msg_final, file=f, flush=True)

    f.close()

'''
python eval_trans_res_memo_cross_vimo.py --dataset_name vimo --vq_name rvq_bs256_finetune_ep10 \
    --name mtrans_memo_cross_l6_bs64_ep200 --res_name rtrans_memo_cross_l6_bs64_ep200 \
    --gpu_id 1 --cond_scale 4 --time_steps 10 --ext rvq1_rtrans1_bs64_cs4_ts10-newData \
    --which_epoch all --test_txt test.txt
'''