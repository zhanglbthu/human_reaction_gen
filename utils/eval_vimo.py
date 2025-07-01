import os

# import clip
import numpy as np
import torch
# from scipy import linalg
from utils.metrics import *
import torch.nn.functional as F
# import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric
#
#
# def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
#     xyz = xyz[:1]
#     bs, seq = xyz.shape[:2]
#     xyz = xyz.reshape(bs, seq, -1, 3)
#     plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
#     plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
#     writer.add_video(tag, plot_xyz, nb_iter, fps=20)


@torch.no_grad()
def evaluation_vqvae_vimo_in_train(out_dir, val_loader, net, writer, ep, best_fid, best_div, eval_wrapper, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    nb_sample = 0
    for batch in val_loader:
        # print(len(batch))
        motion, m_length = batch

        motion = motion.cuda()
        em = eval_wrapper.get_motion_embeddings(motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)

        em_pred = eval_wrapper.get_motion_embeddings(pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f"%\
          (ep, fid, diversity_real, diversity)
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!! Diversity_Real: %.5f"%(best_div, diversity, diversity_real)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_div.tar'))

    # if save:
    #     torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_last.tar'))

    net.train()
    return best_fid, best_div, writer

@torch.no_grad()
def evaluation_vqvae_plus_mpjpe_vimo_in_train(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_mpjpe, eval_wrapper, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    nb_sample = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        # print(len(batch))
        motion, m_length = batch

        motion = motion.cuda()
        em = eval_wrapper.get_motion_embeddings(motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        em_pred = eval_wrapper.get_motion_embeddings(pred_pose_eval, m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joints)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joints)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]

        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, MPJPE. %.4f" % \
          (ep, fid, diversity_real, diversity, mpjpe)
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/MPJPE', mpjpe, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!! Diversity_Real: %.5f"%(best_div, diversity, diversity_real)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_div.tar'))

    if mpjpe < best_mpjpe:
        msg = "--> --> \t MPJPE Improved from %.5f to %.5f !!!" % (best_mpjpe, mpjpe)
        if draw: print(msg)
        best_mpjpe = mpjpe
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mpjpe.tar'))

    # if save:
    #     torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_last.tar'))

    net.train()
    return best_fid, best_div, best_mpjpe, writer

@torch.no_grad()
def evaluation_vqvae_plus_mpjpe_vimo(val_loader, net, repeat_id, eval_wrapper, num_joints):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    nb_sample = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        # print(len(batch))
        motion, m_length = batch

        motion = motion.cuda()
        em = eval_wrapper.get_motion_embeddings(motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        em_pred = eval_wrapper.get_motion_embeddings(pred_pose_eval, m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joints)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joints)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]

        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, MPJPE. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, mpjpe)
    # logger.info(msg)
    print(msg)
    return fid, diversity_real, diversity, mpjpe

@torch.no_grad()
def evaluation_mask_transformer(out_dir, val_loader, trans, vq_model, video_encoder, writer, ep, best_fid, best_div,
                           eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()
    video_encoder.eval()

    motion_annotation_list = []
    motion_pred_list = []

    time_steps = 10  # change 18 to 10 according to https://github.com/EricGuo5513/momask-codes/issues/57
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        imgs, pose, m_length, video_path = batch
        imgs_feat_mean, _ = video_encoder(imgs.cuda())
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # (b, seqlen)
        mids = trans.generate(imgs_feat_mean, m_length//4, time_steps, cond_scale, temperature=1)

        # motion_codes = motion_codes.permute(0, 2, 1)
        mids.unsqueeze_(-1)
        pred_motions = vq_model.forward_decoder(mids)

        em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = ['' for _ in rand_idx]  # [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, writer

@torch.no_grad()
def evaluation_res_transformer(out_dir, val_loader, trans, vq_model, video_encoder, writer, ep, best_fid, best_div,
                           eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, cond_scale=2, temperature=1):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()
    video_encoder.eval()

    motion_annotation_list = []
    motion_pred_list = []

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        imgs, pose, m_length, video_path = batch
        imgs_feat_mean, _ = video_encoder(imgs.cuda())
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # (b, seqlen)
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            pred_ids = trans.generate(code_indices[..., 0], imgs_feat_mean, m_length//4,
                                      temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = ['' for _ in rand_idx]  # [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, writer

@torch.no_grad()
def evaluation_mask_transformer_test_plus_res(val_loader, vq_model, res_model, trans, video_encoder, 
                                              repeat_id, eval_wrapper, time_steps,
                                              cond_scale, temperature, topkr, gsample=True, 
                                              force_mask=False, cal_mm=True, res_cond_scale=5):
                                              
    trans.eval()
    vq_model.eval()
    res_model.eval()
    video_encoder.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        imgs, pose, m_length, video_path = batch
        imgs_feat_mean, _ = video_encoder(imgs.cuda())
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(imgs_feat_mean, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                # mids.unsqueeze_(-1)
                pred_ids = res_model.generate(mids, imgs_feat_mean, m_length // 4, temperature=1, cond_scale=res_cond_scale)
                # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
                # pred_ids = torch.where(pred_ids==-1, 0, pred_ids)

                pred_motions = vq_model.forward_decoder(pred_ids)

                # pred_motions = vq_model.decoder(codes)
                # pred_motions = vq_model.forward_decoder(mids)

                em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(imgs_feat_mean, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            # mids.unsqueeze_(-1)
            pred_ids = res_model.generate(mids, imgs_feat_mean, m_length // 4, temperature=1, cond_scale=res_cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
            # pred_ids = torch.where(pred_ids == -1, 0, pred_ids)

            pred_motions = vq_model.forward_decoder(pred_ids)
            # pred_motions = vq_model.forward_decoder(mids)

            em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity_real, diversity, multimodality

@torch.no_grad()
def evaluation_mask_transformer_test(val_loader, vq_model, trans, video_encoder, 
                                              repeat_id, eval_wrapper, time_steps,
                                              cond_scale, temperature, topkr, gsample=True, 
                                              force_mask=False, cal_mm=True, res_cond_scale=5):
                                              
    trans.eval()
    vq_model.eval()
    video_encoder.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        imgs, pose, m_length, video_path = batch
        imgs_feat_mean, _ = video_encoder(imgs.cuda())
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(imgs_feat_mean, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                mids.unsqueeze_(-1)
                pred_motions = vq_model.forward_decoder(mids)

                em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(imgs_feat_mean, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            mids.unsqueeze_(-1)
            pred_motions = vq_model.forward_decoder(mids)

            em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity_real, diversity, multimodality

@torch.no_grad()
def evaluation_mask_transformer_fe(out_dir, val_loader, trans, vq_model, video_encoder, fer, fuser, writer, ep, best_fid, best_div,
                           eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        fuser_state_dict = fuser.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'fuser': fuser_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()
    video_encoder.eval()
    fer.model.eval()
    fuser.eval()

    motion_annotation_list = []
    motion_pred_list = []

    time_steps = 10  # change 18 to 10 according to https://github.com/EricGuo5513/momask-codes/issues/57
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        imgs, pose, m_length, video_path = batch
        at_features_mean, _ = video_encoder(imgs.cuda())
        middle_frame = imgs[:, imgs.shape[1]//2, :, :, :]
        fe_features = fer.model(middle_frame.cuda())
        imgs_feat = fuser(fe_features, at_features_mean)

        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # (b, seqlen)
        mids = trans.generate(imgs_feat, m_length//4, time_steps, cond_scale, temperature=1)

        # motion_codes = motion_codes.permute(0, 2, 1)
        mids.unsqueeze_(-1)
        pred_motions = vq_model.forward_decoder(mids)

        em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = ['' for _ in rand_idx]  # [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, writer

@torch.no_grad()
def evaluation_mask_transformer_fe_memo(out_dir, val_loader, trans, vq_model, video_encoder, fer, fuser, writer, ep, best_fid, best_div,
                           eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        fuser_state_dict = fuser.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'fuser': fuser_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()
    video_encoder.eval()
    fer.model.eval()
    fuser.eval()

    motion_annotation_list = []
    motion_pred_list = []

    time_steps = 10  # change 18 to 10 according to https://github.com/EricGuo5513/momask-codes/issues/57
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        imgs, pose, m_length, video_path = batch
        at_features_mean, at_features = video_encoder(imgs.cuda())
        middle_frame = imgs[:, imgs.shape[1]//2, :, :, :]
        fe_features = fer.model(middle_frame.cuda())
        imgs_feat = fuser(fe_features, at_features_mean)

        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # (b, seqlen)
        mids = trans.generate(imgs_feat, m_length//4, time_steps, cond_scale, temperature=1, memory=at_features)

        # motion_codes = motion_codes.permute(0, 2, 1)
        mids.unsqueeze_(-1)
        pred_motions = vq_model.forward_decoder(mids)

        em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = ['' for _ in rand_idx]  # [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, writer

@torch.no_grad()
def evaluation_mask_transformer_memo(out_dir, val_loader, trans, vq_model, video_encoder, writer, ep, best_fid, best_div,
                           eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()
    video_encoder.eval()

    motion_annotation_list = []
    motion_pred_list = []

    time_steps = 10  # change 18 to 10 according to https://github.com/EricGuo5513/momask-codes/issues/57
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        imgs, pose, m_length, video_path = batch
        at_features_mean, at_features = video_encoder(imgs.cuda())

        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # (b, seqlen)
        mids = trans.generate(at_features_mean, m_length//4, time_steps, cond_scale, temperature=1, memory=at_features)

        # motion_codes = motion_codes.permute(0, 2, 1)
        mids.unsqueeze_(-1)
        pred_motions = vq_model.forward_decoder(mids)

        em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = ['' for _ in rand_idx]  # [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, writer

@torch.no_grad()
def evaluation_res_transformer_fe(out_dir, val_loader, trans, vq_model, video_encoder, fer, fuser, writer, ep, best_fid, best_div,
                           eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, cond_scale=2, temperature=1):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        fuser_state_dict = fuser.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            'fuser': fuser_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()
    video_encoder.eval()
    fer.model.eval()
    fuser.eval()

    motion_annotation_list = []
    motion_pred_list = []

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        imgs, pose, m_length, video_path = batch
        at_features_mean, _ = video_encoder(imgs.cuda())
        middle_frame = imgs[:, imgs.shape[1]//2, :, :, :]
        fe_features = fer.model(middle_frame.cuda())
        imgs_feat = fuser(fe_features, at_features_mean)

        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # (b, seqlen)
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            pred_ids = trans.generate(code_indices[..., 0], imgs_feat, m_length//4,
                                      temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = ['' for _ in rand_idx]  # [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, writer

@torch.no_grad()
def evaluation_res_transformer_fe_memo(out_dir, val_loader, trans, vq_model, video_encoder, fer, fuser, writer, ep, best_fid, best_div,
                           eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, cond_scale=2, temperature=1):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        fuser_state_dict = fuser.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            'fuser': fuser_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()
    video_encoder.eval()
    fer.model.eval()
    fuser.eval()

    motion_annotation_list = []
    motion_pred_list = []

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        imgs, pose, m_length, video_path = batch
        at_features_mean, at_features = video_encoder(imgs.cuda())
        middle_frame = imgs[:, imgs.shape[1]//2, :, :, :]
        fe_features = fer.model(middle_frame.cuda())
        imgs_feat = fuser(fe_features, at_features_mean)

        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # (b, seqlen)
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            pred_ids = trans.generate(code_indices[..., 0], imgs_feat, m_length//4,
                                      temperature=temperature, cond_scale=cond_scale, memory=at_features)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = ['' for _ in rand_idx]  # [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, writer

@torch.no_grad()
def evaluation_res_transformer_memo(out_dir, val_loader, trans, vq_model, video_encoder, writer, ep, best_fid, best_div,
                           eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, cond_scale=2, temperature=1):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()
    video_encoder.eval()

    motion_annotation_list = []
    motion_pred_list = []

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        imgs, pose, m_length, video_path = batch
        at_features_mean, at_features = video_encoder(imgs.cuda())

        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # (b, seqlen)
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            pred_ids = trans.generate(code_indices[..., 0], at_features_mean, m_length//4,
                                      temperature=temperature, cond_scale=cond_scale, memory=at_features)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = ['' for _ in rand_idx]  # [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, writer

@torch.no_grad()
def evaluation_mask_transformer_test_plus_res_fe(val_loader, vq_model, res_model, trans, video_encoder, fer, trans_fuser, res_fuser,
                                              repeat_id, eval_wrapper, time_steps,
                                              cond_scale, temperature, topkr, gsample=True, 
                                              force_mask=False, cal_mm=True, res_cond_scale=5):
                                              
    trans.eval()
    vq_model.eval()
    res_model.eval()
    video_encoder.eval()
    fer.model.eval()
    trans_fuser.eval()
    res_fuser.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        imgs, pose, m_length, video_path = batch
        at_features_mean, _ = video_encoder(imgs.cuda())
        middle_frame = imgs[:, imgs.shape[1]//2, :, :, :]
        fe_features = fer.model(middle_frame.cuda())
        imgs_feat4trans = trans_fuser(fe_features, at_features_mean)
        imgs_feat4res = res_fuser(fe_features, at_features_mean)

        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(imgs_feat4trans, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                # mids.unsqueeze_(-1)
                pred_ids = res_model.generate(mids, imgs_feat4res, m_length // 4, temperature=1, cond_scale=res_cond_scale)
                # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
                # pred_ids = torch.where(pred_ids==-1, 0, pred_ids)

                pred_motions = vq_model.forward_decoder(pred_ids)

                # pred_motions = vq_model.decoder(codes)
                # pred_motions = vq_model.forward_decoder(mids)

                em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(imgs_feat4trans, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            # mids.unsqueeze_(-1)
            pred_ids = res_model.generate(mids, imgs_feat4res, m_length // 4, temperature=1, cond_scale=res_cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
            # pred_ids = torch.where(pred_ids == -1, 0, pred_ids)

            pred_motions = vq_model.forward_decoder(pred_ids)
            # pred_motions = vq_model.forward_decoder(mids)

            em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity_real, diversity, multimodality

@torch.no_grad()
def evaluation_mask_transformer_test_plus_res_fe_memo(val_loader, vq_model, res_model, trans, video_encoder, fer, trans_fuser, res_fuser,
                                              repeat_id, eval_wrapper, time_steps,
                                              cond_scale, temperature, topkr, gsample=True, 
                                              force_mask=False, cal_mm=True, res_cond_scale=5):
                                              
    trans.eval()
    vq_model.eval()
    res_model.eval()
    video_encoder.eval()
    fer.model.eval()
    trans_fuser.eval()
    res_fuser.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        imgs, pose, m_length, video_path = batch
        at_features_mean, at_features = video_encoder(imgs.cuda())
        middle_frame = imgs[:, imgs.shape[1]//2, :, :, :]
        fe_features = fer.model(middle_frame.cuda())
        imgs_feat4trans = trans_fuser(fe_features, at_features_mean)
        imgs_feat4res = res_fuser(fe_features, at_features_mean)

        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(imgs_feat4trans, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask, memory=at_features)

                # motion_codes = motion_codes.permute(0, 2, 1)
                # mids.unsqueeze_(-1)
                pred_ids = res_model.generate(mids, imgs_feat4res, m_length // 4, temperature=1, cond_scale=res_cond_scale, memory=at_features)
                # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
                # pred_ids = torch.where(pred_ids==-1, 0, pred_ids)

                pred_motions = vq_model.forward_decoder(pred_ids)

                # pred_motions = vq_model.decoder(codes)
                # pred_motions = vq_model.forward_decoder(mids)

                em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(imgs_feat4trans, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask, memory=at_features)

            # motion_codes = motion_codes.permute(0, 2, 1)
            # mids.unsqueeze_(-1)
            pred_ids = res_model.generate(mids, imgs_feat4res, m_length // 4, temperature=1, cond_scale=res_cond_scale, memory=at_features)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
            # pred_ids = torch.where(pred_ids == -1, 0, pred_ids)

            pred_motions = vq_model.forward_decoder(pred_ids)
            # pred_motions = vq_model.forward_decoder(mids)

            em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity_real, diversity, multimodality

@torch.no_grad()
def evaluation_mask_transformer_test_plus_res_memo(val_loader, vq_model, res_model, trans, video_encoder,
                                              repeat_id, eval_wrapper, time_steps,
                                              cond_scale, temperature, topkr, gsample=True, 
                                              force_mask=False, cal_mm=True, res_cond_scale=5):
                                              
    trans.eval()
    vq_model.eval()
    res_model.eval()
    video_encoder.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        imgs, pose, m_length, video_path = batch
        at_features_mean, at_features = video_encoder(imgs.cuda())

        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(at_features_mean, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask, memory=at_features)

                # motion_codes = motion_codes.permute(0, 2, 1)
                # mids.unsqueeze_(-1)
                pred_ids = res_model.generate(mids, at_features_mean, m_length // 4, temperature=1, cond_scale=res_cond_scale, memory=at_features)
                # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
                # pred_ids = torch.where(pred_ids==-1, 0, pred_ids)

                pred_motions = vq_model.forward_decoder(pred_ids)

                # pred_motions = vq_model.decoder(codes)
                # pred_motions = vq_model.forward_decoder(mids)

                em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(at_features_mean, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask, memory=at_features)

            # motion_codes = motion_codes.permute(0, 2, 1)
            # mids.unsqueeze_(-1)
            pred_ids = res_model.generate(mids, at_features_mean, m_length // 4, temperature=1, cond_scale=res_cond_scale, memory=at_features)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
            # pred_ids = torch.where(pred_ids == -1, 0, pred_ids)

            pred_motions = vq_model.forward_decoder(pred_ids)
            # pred_motions = vq_model.forward_decoder(mids)

            em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 13) # 100
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 13) # 100
 
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity_real, diversity, multimodality

@torch.no_grad()
def evaluation_mask_transformer_fe_test(val_loader, vq_model, trans, video_encoder, fer, trans_fuser,
                                              repeat_id, eval_wrapper, time_steps,
                                              cond_scale, temperature, topkr, gsample=True, 
                                              force_mask=False, cal_mm=True):
                                              
    trans.eval()
    vq_model.eval()
    video_encoder.eval()
    fer.model.eval()
    trans_fuser.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        imgs, pose, m_length, video_path = batch
        at_features_mean, _ = video_encoder(imgs.cuda())
        middle_frame = imgs[:, imgs.shape[1]//2, :, :, :]
        fe_features = fer.model(middle_frame.cuda())
        imgs_feat4trans = trans_fuser(fe_features, at_features_mean)

        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(imgs_feat4trans, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                mids.unsqueeze_(-1)
                pred_motions = vq_model.forward_decoder(mids)

                em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(imgs_feat4trans, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            mids.unsqueeze_(-1)
            pred_motions = vq_model.forward_decoder(mids)

            em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity_real, diversity, multimodality