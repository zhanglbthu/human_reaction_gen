import os
from os.path import join as pjoin

import torch
import torch.nn.functional as F

from models.mask_transformer.transformer_memo_cross import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator
from models.tc_clip.tc_clip import TCCLIP_VE
from custom_clip import clip

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical
from data.vimo_dataset import test_pipeline

from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

clip_version = 'ViT-B/16'

def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                vq_opt.dim_pose,
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
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, opt, which_model):
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
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]} Completed!')
    return t2m_transformer

def load_res_model(res_opt, vq_opt, opt):
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

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]} Completed!')
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
                    "temporal_length": 16,
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

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 263
    
    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./generation', opt.ext)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    #######################
    ######Loading RVQ######
    #######################
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading R-Transformer######
    #################################
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt)

    assert res_opt.vq_name == model_opt.vq_name

    #################################
    ######Loading M-Transformer######
    #################################
    t2m_transformer = load_trans_model(model_opt, opt, 'net_best_fid.tar')  # 'latest.tar'

    ##################################
    #####Loading Video Encoder########
    ##################################
    video_encoder = prepare_video_encoder(clip_version)

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    video_encoder.eval()

    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)
    video_encoder.to(opt.device)
    
    ##### ---- Dataloader ---- #####
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))

    def inv_transform(data):
        return data * std + mean
    
    opt.nb_joints = 22

    from data.pipeline import Compose
    data_pipeline = Compose(test_pipeline)

    video_list = []
    video_path_list = []
    length_list = []

    print('Preprocessing video data...')
    if opt.video_path != "":
        video_info = dict(filename=opt.video_path, tar=False, start_index=0, modality='RGB')
        results = data_pipeline(video_info)
        video_list.append(results['imgs'])
        video_path_list.append(opt.video_path)
        if opt.motion_length == 0:
            length_list.append(88)
        else:
            length_list.append(opt.motion_length)
    elif opt.video_path_file != "":
        with open(opt.vide_opath_file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                infos = line.strip().split('#')
                video_info = dict(filename=infos[0], tar=False, start_index=0, modality='RGB')
                results = data_pipeline(video_info)
                video_list.append(results['imgs'])
                video_path_list.append(infos[0])
                if len(infos) == 1 or (not infos[1].isdigit()):
                    length_list.append(88)
                else:
                    length_list.append(int(infos[-1]))
    else:
        raise "A video path, or a file of video paths are required!!!"

    token_lens = torch.LongTensor(length_list) // 4
    token_lens = token_lens.to(opt.device).long()
    m_length = token_lens * 4

    imgs = torch.stack(video_list).to(opt.device)
    print('Extracting image features...')
    at_features_mean, at_features = video_encoder(imgs)

    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():
            mids = t2m_transformer.generate(at_features_mean, token_lens,
                                            timesteps=opt.time_steps,
                                            cond_scale=opt.cond_scale,
                                            temperature=opt.temperature,
                                            topk_filter_thres=opt.topkr,
                                            gsample=opt.gumbel_sample,
                                            memory=at_features)
            # print(mids)
            # print(mids.shape)
            mids = res_model.generate(mids, at_features_mean, token_lens, temperature=1, cond_scale=5, memory=at_features)
            pred_motions = vq_model.forward_decoder(mids)

            pred_motions = pred_motions.detach().cpu().numpy()

            data = inv_transform(pred_motions)
        print('Saving results...')
        for k, (video_path, joint_data) in enumerate(zip(video_path_list, data)):
            print("---->Sample %d: %s %d"%(k, video_path, m_length[k]))
            video_id = video_path.replace('\\', '/').split('/')[-1][:-4]
            animation_path = pjoin(animation_dir, video_id)
            joint_path = pjoin(joints_dir, video_id)

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.bvh"%(k, r, m_length[k]))
            _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
            _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)


            save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4"%(k, r, m_length[k]))
            ik_save_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.mp4"%(k, r, m_length[k]))

            plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title='', fps=20)
            plot_3d_motion(save_path, kinematic_chain, joint, title='', fps=20)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])), joint)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_ik.npy"%(k, r, m_length[k])), ik_joint)
            