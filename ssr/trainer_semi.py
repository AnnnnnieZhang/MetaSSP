"""Stage 2 of MetaSSP: semi-supervised teacher-student training.

Per epoch:
  1. estimate the per-parameter importance of the teacher on unlabeled data;
  2. for every (labeled, unlabeled) batch pair, train the student on the supervised
     loss blended with the pseudo-label loss, weighted by the SDF-aware pseudo-label
     weight, and update the teacher with the importance-regularised EMA;
  3. evaluate student and teacher, save checkpoints, let the meta-controller rescale
     the EMA momentum and, if needed, apply a "reset" EMA update.
"""
import os
import csv
import time
import datetime
import itertools

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.rend_util import get_psnr
from metassp.ema import ema_update, MomentumController
from metassp.importance import compute_importance_weights
from metassp.augmentation import batch_strong_augmentation, batch_weak_augmentation
from metassp.pseudo_label import pseudo_label_weight, blend_losses

LOSS_KEYS = ['rgb_loss', 'eikonal_loss', 'depth_loss', 'normal_l1', 'normal_cos',
             'ray_mask_loss', 'instance_mask_loss', 'sdf_loss', 'vis_sdf_loss']


def _to_device(batch):
    return {k: v.cuda().to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _detach(outputs):
    return {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}


def _format_losses(prefix, start_t, epoch, batch_id, num_batches, loss_output, psnr, net):
    beta = net.module.density.get_beta().item()
    terms = ', '.join(f'{k} = {loss_output[k].item()}' for k in LOSS_KEYS)
    elapsed = str(datetime.timedelta(seconds=round(time.time() - start_t)))
    return (f'{prefix}{elapsed:0>8},[epoch {epoch}] ({batch_id + 1}/{num_batches}): '
            f'total_loss = {loss_output["total_loss"].item()}, {terms}, '
            f'psnr = {psnr.item()}, beta={beta}, alpha={1. / beta}')


def _accumulate(info, loss_output):
    for key in loss_output:
        if "total" not in key:
            info[key] = info.get(key, 0) + torch.mean(loss_output[key]).item()


def Recon_trainer(cfg, model, tea_net, loss, optimizer, scheduler, train_loader, pseudo_train_loader,
                  test_loader, device, checkpoint, tea_checkpoint):
    start_t = time.time()
    config = cfg.config
    stage2 = config['Stage2']
    nepoch = config['other']['nepoch']

    log_dir = cfg.save_path
    os.makedirs(log_dir, exist_ok=True)
    cfg.write_config()
    tb_logger = SummaryWriter(log_dir)

    # Per-epoch evaluation trend and per-batch pseudo-label weights.
    trend_file_path = os.path.join(log_dir, 'trend_stage2.csv')
    weight_file_path = os.path.join(log_dir, 'weight_stage2.csv')
    if not os.path.exists(trend_file_path):
        with open(trend_file_path, 'w', newline='') as f:
            csv.writer(f).writerow(["Epoch", "AvgEvalLoss", "AvgSDFLoss", "TeaAvgEvalLoss",
                                    "TeaAvgSDFLoss", "ema_alpha", "ema_eta", "Stu->Tea"])
    if not os.path.exists(weight_file_path):
        with open(weight_file_path, 'w', newline='') as f:
            csv.writer(f).writerow(["batch", "tea_loss", "sdf", "pseudo_W"])

    start_epoch = 0
    iter = 0
    if config["resume"] == True:
        checkpoint.load(config["weight"])
        start_epoch = checkpoint.module_dict['epoch']
        iter = checkpoint.module_dict['iter']
    else:
        # The student starts as a copy of the warm-up teacher.
        model.load_state_dict(tea_net.state_dict())
    if config['finetune'] == True:
        start_epoch = 0
    scheduler.last_epoch = start_epoch

    tea_net = tea_net.to(device)
    tea_net.eval()

    min_eval_loss = 10000
    min_eval_loss_teacher = 10000

    accumulation_steps = config["data"]["accumulation_steps"]
    min_len = min(len(train_loader), len(pseudo_train_loader))
    total_steps = nepoch * min_len

    # EMA settings (paper Sec. III-D)
    momentum = stage2['ema_alpha']                              # m_0 of the cosine schedule; rescaled by the
                                                                # meta-controller and clamped at every evaluation
    momentum_min = stage2.get('ema_min', 0.96)
    momentum_max = stage2.get('ema_max', 0.9996)
    eta = stage2['eta']
    importance_batches = stage2.get('importance_batches', 100)
    controller = MomentumController(hidden=stage2.get('mc_hidden', 16)).to(device)

    # "Reset" EMA update when the student stops being worse than the teacher.
    reset_interval = stage2.get('reset_interval', 3)
    reset_scale = stage2.get('reset_momentum_scale', 0.6)
    reset_switch_epoch = stage2.get('reset_switch_epoch', 70)
    reset_delta_sdf = stage2.get('reset_delta_sdf', 0.002)      # before reset_switch_epoch: SDF-loss gap
    reset_delta_total = stage2.get('reset_delta_total', 0.004)  # afterwards: total-loss gap

    # Pseudo-label weighting (paper Eq. 4-5)
    pseudo_alpha = stage2.get('pseudo_alpha', 4.0)
    pseudo_beta = stage2.get('pseudo_beta', 4.0)
    pseudo_lambda = stage2.get('pseudo_lambda', 0.2)

    for e in range(start_epoch, nepoch):
        torch.cuda.empty_cache()
        cfg.log_string("Switch Phase to Train")
        model.train()
        optimizer.zero_grad()

        # Gradient-based parameter importance of the current teacher (paper Sec. III-C).
        importance_weights = compute_importance_weights(tea_net, pseudo_train_loader, device,
                                                        num_batches=importance_batches)

        for batch_id, ((indices, model_input, ground_truth), (pseudo_indices, pseudo_model_input)) in enumerate(
            itertools.islice(zip(train_loader, pseudo_train_loader), min_len)
        ):
            '''
            indices: [B*1]
            model_input:
                1.image: [B, 3, H, W]
                2.uv: [B, H*W, 2]                           image coordinate(pixel)
                3.intrinsics: [B, 3, 3]                     image to camera
                4.pose: [B, 4, 4]                           camera to world
                5*.add_points_world: [B, N_uv, N_add, 3]    add points in world coords
            ground_truth:
                1.rgb: [B, num_pixels, 3]
                2.depth: [B, num_pixels, 1]
                3.normal: [B, num_pixels, 3]
            pseudo_model_input: same as model_input, without ground truth
            '''
            model_input = _to_device(model_input)
            pseudo_model_input = _to_device(pseudo_model_input)
            ground_truth = _to_device(ground_truth)

            # Teacher pseudo-labels on weakly augmented unlabeled images. The SDF network
            # needs autograd for its normals, so outputs are detached instead of using no_grad.
            img_o = pseudo_model_input["image"]
            pseudo_model_input["image"] = batch_weak_augmentation(img_o)
            teacher_outputs = _detach(tea_net(pseudo_model_input, pseudo_indices))

            # Student: supervised loss on labeled data.
            model_outputs = model(model_input, indices)
            supervised_loss = loss(model_outputs, ground_truth, e, label=True)

            # Student: pseudo-label loss on strongly augmented unlabeled images.
            pseudo_model_input["image"] = batch_strong_augmentation(img_o)
            pseudo_model_outputs = model(pseudo_model_input, pseudo_indices)
            pseudo_loss = loss(pseudo_model_outputs, teacher_outputs, e, label=False)

            # Pseudo-label weight: teacher consistency under strong vs. weak augmentation
            # and variance of the weakly augmented SDF (paper Eq. 2-4).
            pseudo_model_input["image"] = batch_strong_augmentation(img_o)
            teacher_outputs_strong = _detach(tea_net(pseudo_model_input, pseudo_indices))
            consistency = loss(teacher_outputs_strong, teacher_outputs, e, label=False)['total_loss']
            sdf_var = teacher_outputs['sdf'].reshape(-1, 1).var()
            pseudo_W = pseudo_label_weight(consistency.item(), sdf_var.item(), pseudo_alpha, pseudo_beta)

            # Blend supervised and pseudo-supervised loss terms (paper Eq. 5).
            loss_output = blend_losses(supervised_loss, pseudo_loss, pseudo_W, pseudo_lambda)
            total_loss = loss_output['total_loss']

            # Gradient accumulation.
            (total_loss / accumulation_steps).backward()

            total_norm = 0
            if (batch_id + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.7, norm_type=2)

                parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                total_norm = sum(p.grad.detach().norm(2).item() ** 2 for p in parameters) ** 0.5

                # Importance-regularised EMA update of the teacher (paper Eq. 10).
                ema_update(tea_net, model,
                           step=batch_id + e * min_len,
                           total_steps=total_steps,
                           initial_momentum=momentum,
                           importance_weights=importance_weights, eta=eta)

                optimizer.step()
                optimizer.zero_grad()

            psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].reshape(-1, 3))
            cfg.log_string(_format_losses('', start_t, e, batch_id, min_len, loss_output, psnr, model))

            tb_logger.add_scalar('Loss/total_loss', total_loss.item(), iter)
            tb_logger.add_scalar('Loss/color_loss', loss_output['rgb_loss'].item(), iter)
            tb_logger.add_scalar('Loss/eikonal_loss', loss_output['eikonal_loss'].item(), iter)
            tb_logger.add_scalar('Loss/smooth_loss', loss_output['smooth_loss'].item(), iter)
            tb_logger.add_scalar('Loss/depth_loss', loss_output['depth_loss'].item(), iter)
            tb_logger.add_scalar('Loss/normal_l1_loss', loss_output['normal_l1'].item(), iter)
            tb_logger.add_scalar('Loss/normal_cos_loss', loss_output['normal_cos'].item(), iter)
            tb_logger.add_scalar('Loss/ray_mask_loss', loss_output['ray_mask_loss'].item(), iter)
            tb_logger.add_scalar('Loss/instance_mask_loss', loss_output['instance_mask_loss'].item(), iter)
            tb_logger.add_scalar('Loss/sdf_loss', loss_output['sdf_loss'].item(), iter)
            tb_logger.add_scalar('Loss/vis_sdf_loss', loss_output['vis_sdf_loss'].item(), iter)
            tb_logger.add_scalar('Loss/grad_norm', total_norm, iter)

            beta = model.module.density.get_beta().item()
            tb_logger.add_scalar('Statistics/beta', beta, iter)
            tb_logger.add_scalar('Statistics/alpha', 1. / beta, iter)
            tb_logger.add_scalar('Statistics/psnr', psnr.item(), iter)
            tb_logger.add_scalar("train/lr", optimizer.state_dict()['param_groups'][0]['lr'], iter)

            tb_logger.add_scalar('train/tea_loss', consistency.item(), iter)
            tb_logger.add_scalar('train/sdf_var', sdf_var.item(), iter)
            tb_logger.add_scalar('train/pseudo_W', pseudo_W, iter)

            iter += 1

            with open(weight_file_path, 'a', newline='') as f:
                csv.writer(f).writerow([batch_id + 1, f"{consistency.item():.6f}",
                                        f"{sdf_var.item():.6f}", f"{pseudo_W:.6f}"])

        scheduler.step()

        # Every model_save_interval epochs: evaluate student and teacher, save checkpoints.
        if e % config['other']['model_save_interval'] != 0:
            continue

        model.eval()
        eval_loss = eval_sdf_loss = 0
        tea_eval_loss = tea_eval_sdf_loss = 0
        eval_loss_info = {}
        tea_eval_loss_info = {}
        student_to_teacher = False

        cfg.log_string("Switch Phase to Test")
        for batch_id, (indices, model_input, ground_truth) in enumerate(test_loader):
            torch.cuda.empty_cache()
            model_input = _to_device(model_input)
            ground_truth = _to_device(ground_truth)

            model_outputs = model(model_input, indices)
            tea_model_outputs = tea_net(model_input, indices)

            loss_output = loss(model_outputs, ground_truth, e)
            tea_loss_output = loss(tea_model_outputs, ground_truth, e)
            total_loss = loss_output['total_loss']
            tea_total_loss = tea_loss_output['total_loss']

            rgb_gt = ground_truth['rgb'].reshape(-1, 3)
            psnr = get_psnr(model_outputs['rgb_values'], rgb_gt)
            tea_psnr = get_psnr(tea_model_outputs['rgb_values'], rgb_gt)

            cfg.log_string(_format_losses('Validation_Student ', start_t, e, batch_id, len(test_loader),
                                          loss_output, psnr, model))
            cfg.log_string(_format_losses('Validation_Teacher ', start_t, e, batch_id, len(test_loader),
                                          tea_loss_output, tea_psnr, tea_net))

            _accumulate(eval_loss_info, loss_output)
            _accumulate(tea_eval_loss_info, tea_loss_output)

            if torch.isnan(total_loss).item():
                print("NaN detected in total_loss!")
            else:
                eval_loss += total_loss.item()
                eval_sdf_loss += loss_output['sdf_loss'].item()

            if torch.isnan(tea_total_loss).item():
                print("NaN detected in tea_total_loss!")
            else:
                tea_eval_loss += tea_total_loss.item()
                tea_eval_sdf_loss += tea_loss_output['sdf_loss'].item()

        num_eval_batches = batch_id + 1
        avg_eval_loss = eval_loss / num_eval_batches
        avg_sdf_loss = eval_sdf_loss / num_eval_batches
        tea_avg_eval_loss = tea_eval_loss / num_eval_batches
        tea_avg_sdf_loss = tea_eval_sdf_loss / num_eval_batches

        cfg.log_string(f'avg_eval_loss is {avg_eval_loss}')
        cfg.log_string(f'avg_sdf_loss is {avg_sdf_loss}')
        cfg.log_string(f'tea_avg_eval_loss is {tea_avg_eval_loss}')
        cfg.log_string(f'tea_avg_sdf_loss is {tea_avg_sdf_loss}')
        tb_logger.add_scalar('eval/eval_loss', avg_eval_loss, e)
        tb_logger.add_scalar('eval/tea_eval_loss', tea_avg_eval_loss, e)
        for key in eval_loss_info:
            tb_logger.add_scalar("eval/" + key, eval_loss_info[key] / num_eval_batches, e)
        for key in tea_eval_loss_info:
            tb_logger.add_scalar("eval/tea_" + key, tea_eval_loss_info[key] / num_eval_batches, e)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_eval_loss)

        # Checkpoints: <log_dir>/train_epoch_<e>/ for the latest, <log_dir>/train_best/ for the best.
        epoch_folder = os.path.join(log_dir, f'train_epoch_{e}')
        best_folder = os.path.join(log_dir, 'train_best')
        os.makedirs(epoch_folder, exist_ok=True)
        os.makedirs(best_folder, exist_ok=True)

        checkpoint.register_modules(epoch=e, iter=iter, min_loss=avg_eval_loss)
        tea_checkpoint.register_modules(epoch=e, iter=iter, min_loss=tea_avg_eval_loss)

        # Both CheckpointIO objects share cfg, so the save path is switched through it.
        cfg.config['log']['path'] = epoch_folder
        checkpoint.save("latest_student")
        tea_checkpoint.save("latest_teacher")
        cfg.log_string(f"Student / teacher saved to {epoch_folder}/model_latest_{{student,teacher}}.pth")

        cfg.config['log']['path'] = best_folder
        if avg_eval_loss < min_eval_loss:
            checkpoint.save('best_student')
            min_eval_loss = avg_eval_loss
            cfg.log_string(f"Best student model updated: {best_folder}/model_best_student.pth")
        if tea_avg_eval_loss < min_eval_loss_teacher:
            tea_checkpoint.save("best_teacher")
            min_eval_loss_teacher = tea_avg_eval_loss
            cfg.log_string(f"Best teacher model updated: {best_folder}/model_best_teacher.pth")
        cfg.config['log']['path'] = log_dir

        # Meta-controller (paper Eq. 8-9): gamma rescales the initial momentum m_0 of the
        # cosine schedule, which is then clamped to [m_min, m_max]. The rescaling accumulates
        # over evaluations; the cosine annealing is applied on top of it in ema_update.
        delta = torch.tensor(tea_avg_sdf_loss - avg_sdf_loss, device=device)
        t_eval = torch.tensor(tea_avg_eval_loss, device=device)
        epoch_norm = torch.tensor(e / nepoch, device=device)
        with torch.no_grad():
            gamma = controller(delta, t_eval, epoch_norm).item()
        momentum = float(max(min(gamma * momentum, momentum_max), momentum_min))

        # "Reset" update with a lower momentum when the student is no longer clearly
        # worse than the teacher, to shrink the teacher-student gap (paper Sec. III-D).
        if e % reset_interval == 0:
            if e >= reset_switch_epoch:
                need_reset = avg_eval_loss - tea_avg_eval_loss < reset_delta_total
            else:
                need_reset = avg_sdf_loss - tea_avg_sdf_loss < reset_delta_sdf
            if need_reset:
                ema_update(tea_net, model, step=e, total_steps=nepoch,
                           initial_momentum=momentum * reset_scale)
                student_to_teacher = True

        with open(trend_file_path, 'a', newline='') as f:
            csv.writer(f).writerow([e, f"{avg_eval_loss:.6f}", f"{avg_sdf_loss:.6f}",
                                    f"{tea_avg_eval_loss:.6f}", f"{tea_avg_sdf_loss:.6f}",
                                    f"{momentum:.6f}", eta, student_to_teacher])
