import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader

from models.vq.model import RVQVAE
from models.vq.vq_trainer import RVQTokenizerTrainer
from options.vq_option import arg_parse
from data.vimo_dataset import VimoMotionDataset
from utils import paramUtil
import numpy as np

from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.fixseed import fixseed
import warnings
warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "1"

def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    opt = arg_parse(True)
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/vq/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == "vimo":
        opt.data_root = './Data/VIMO/'
        opt.motion_dir = pjoin(opt.data_root, 'vector_263')
        opt.joints_num = 22
        dim_pose = 263
        fps = 20
        radius = 4
        kinematic_chain = paramUtil.t2m_kinematic_chain
        dataset_opt_path = './checkpoints/vimo/Comp_v6_KLD005/opt.txt'
    else:
        raise KeyError('Dataset Does not Exists')

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    #breakpoint()
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'motions_train.txt')
    val_split_file = pjoin(opt.data_root, 'motions_test.txt')

    net = RVQVAE(opt,
                dim_pose,
                opt.nb_code,
                opt.code_dim,
                opt.code_dim,
                opt.down_t,
                opt.stride_t,
                opt.width,
                opt.depth,
                opt.dilation_growth_rate,
                opt.vq_act,
                opt.vq_norm)

    pc_vq = sum(param.numel() for param in net.parameters())
    #print(net)
    # print("Total parameters of discriminator net: {}".format(pc_vq))
    # all_params += pc_vq_dis

    print('Total parameters of the models: {}M'.format(pc_vq/1000_000))

    trainer = RVQTokenizerTrainer(opt, vq_model=net)

    train_dataset = VimoMotionDataset(opt, mean, std, train_split_file)
    val_dataset = VimoMotionDataset(opt, mean, std, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=8,
                            shuffle=True, pin_memory=True)
    
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'motions_test', device=opt.device)
    #breakpoint()
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper, plot_t2m)

'''
python train_vq_vimo.py --name rvq_bs256_finetune_ep10 --gpu_id 0 --window_size 20 \
    --dataset_name vimo --batch_size 256 --num_quantizers 6 --max_epoch 10 \
    --warm_up_iter 20 --milestones 1600 3200 --finetune
'''