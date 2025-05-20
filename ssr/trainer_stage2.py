import os
import time
import datetime
import itertools
import math

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.rend_util import get_psnr
from SemiSupervised.EMA_update import ema_update, clean_increment_ema, MomentumController
from SSP3D.discriminatorPointsSDF import SDFDiscriminatorWithSurfacePoints
import csv
from collections import OrderedDict
from SemiSupervised.img_enhance import batch_strong_augmentation, batch_weak_augmentation
from utils.sdf_util import get_sdf_gt_worldcoords
from SemiSupervised.importance_utils import compute_importance_weights
from ssr.ssr_utils.utils import  get_model

def Recon_trainer(cfg,model,tea_net,loss,optimizer,scheduler,train_loader,pseudo_train_loader,test_loader,device,checkpoint,tea_checkpoint):
    start_t = time.time()
    config = cfg.config

    log_dir = cfg.save_path
    os.makedirs(log_dir, exist_ok=True)

    tea_net.eval()  
    # for p in tea_net.parameters(): 
    #     p.requires_grad=False


    cfg.write_config()
    tb_logger = SummaryWriter(log_dir)

    # 定义保存 trend.txt 的路径
    trend_file_path = os.path.join(log_dir, 'trend_stage2.csv')
    weight_file_path = os.path.join(log_dir, 'weight_stage2.csv')

    # 确保文件存在（如果不存在则创建） 写入标题行
    if not os.path.exists(trend_file_path):
        with open(trend_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "AvgEvalLoss", "AvgSDFLoss", "TeaAvgEvalLoss",
                            "TeaAvgSDFLoss","ema_alpha","ema_eta","Stu->Tea"])
    if not os.path.exists(weight_file_path):
        with open(weight_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["batch", "tea_loss", "sdf", "pseudo_W"])

    start_epoch = 0
    iter = 0
    tea_last_avg_loss = 10000
    tea_last_avg_sdf = 10000
    
    if config["resume"] == True:
        checkpoint.load(config["weight"])
        start_epoch = checkpoint.module_dict['epoch']
        iter = checkpoint.module_dict['iter']
    else:
        model.load_state_dict(tea_net.state_dict())  # 将 tea_net 的参数复制到 model 中
        # checkpoint.load(config["Stage2"]["teacher_weight"])

    if config['finetune']==True:
        start_epoch=0
    scheduler.last_epoch = start_epoch

    model.train()

    min_eval_loss = 10000
    min_eval_loss_teacher = 10000  # 初始化教师模型最优验证损失

    accumulation_steps = config["data"]["accumulation_steps"]  # 累积步数，尝试不同的值

    min_len = min(len(train_loader), len(pseudo_train_loader))

    initial_momentum=config['Stage2']['ema_alpha']
    
    w_fixed = config['Stage2'].get('w_fixed', 0.5)

    # Initialize Meta-Controller
    controller = MomentumController(hidden=config['Stage2'].get('mc_hidden', 16)).to(device)


    for e in range(start_epoch, config['other']['nepoch']):
        torch.cuda.empty_cache()
        cfg.log_string("Switch Phase to Train")
        model.train()
        optimizer.zero_grad()   # 重置优化器梯度

        # 可选：每个 epoch 开始时计算参数重要性（例如只使用前 10 个 batch，无标签数据）
        importance_weights = compute_importance_weights(tea_net, pseudo_train_loader, device, num_batches=100)
            
        for batch_id, ((indices, model_input, ground_truth), (pseudo_indices, pseudo_model_input)) in enumerate(
            itertools.islice(zip(train_loader, pseudo_train_loader), min_len)
        ):
            pseudo_batch = config['data']['batch_size']['pseudo']
            '''
            indices: [B*1]
            model_input: 
                1.image: [B, 3, H, W]
                2.uv: [B, H*W, 2]                           image coordinate(pixel)
                3.intrinsics: [B, 3, 3]                     image to camera
                4.pose: [B, 4, 4]                           camera to world
                5*.add_points_world: [B, N_uv, N_add, 3]     add points in world coords
            ground_truth:
                1.rgb: [B, num_pixels, 3]
                2.depth: [B, num_pixels, 1]
                3.normal: [B, num_pixels, 3]
            '''

            # 确保所有数据在 GPU 上
            model_input = {k: v.cuda().to(torch.float32) for k, v in model_input.items()}
            pseudo_model_input = {k: v.cuda().to(torch.float32) for k, v in pseudo_model_input.items()}
            ground_truth = {k: v.cuda().to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in ground_truth.items()}

            tea_net = tea_net.to(device)
                
            # 推理模式下运行教师模型

            img_o = pseudo_model_input["image"]
            pseudo_model_input["image"] = batch_weak_augmentation(img_o)
            teacher_outputs = tea_net(pseudo_model_input, pseudo_indices)
            # teacher_outputs_fdist = tea_net(pseudo_model_input, pseudo_indices, f_dist=True)
            # teacher_outputs = {k: w_fixed * out_slow[k] + (1-w_fixed) * out_fast[k] for k in out_slow}
            # 过滤伪标签 ……
            # 获得伪标签的 SDF
                

            # 学生模型监督学习
            model_outputs = model(model_input, indices)
            supervised_loss = loss(model_outputs, ground_truth, e, label=True)

            # 学生模型无监督学习（伪标签损失）
            pseudo_model_input["image"] = batch_strong_augmentation(img_o)
            pseudo_model_outputs = model(pseudo_model_input, pseudo_indices)
            pseudo_loss = loss(pseudo_model_outputs, teacher_outputs, e, label=False)

            # 伪标签权重计算
            pseudo_model_input["image"] = batch_strong_augmentation(img_o)
            teacher_outputs_weight = tea_net(pseudo_model_input, pseudo_indices)
            # 伪标签权重决定因素1 = 原图 和 强加强的 loss
            weight_loss = loss(teacher_outputs_weight, teacher_outputs, e, label=False)
            weight = weight_loss['total_loss']
            # 伪标签权重决定因素2 = sdf 绝对值
            tea_sdf = teacher_outputs['sdf'].reshape(-1, 1)    
            sdf_var = tea_sdf.var() 
            # 伪标签权重决定因素3 = 特征扰动
            # fist_loss = loss(teacher_outputs_fdist, teacher_outputs, e, label=False)
            # fist = fist_loss['total_loss']
            

            # pseudo_W = 1-(weight.item()/2 + sdf_var.item())/2*10
            pseudo_W = 1-(weight.item() + sdf_var.item() )*4
            pseudo_W = max(0, min(pseudo_W, 1))

            # W = compute_pseudo_quality(weight, tea_sdf, combine_method='product')
            # print("伪标签质量 W =", W.item())



            # 加权融合损失字典
            loss_weight_pseudo = pseudo_W*0.1
            loss_output = {
                key: (1 - loss_weight_pseudo) * supervised_loss[key]
                    + loss_weight_pseudo * pseudo_loss[key]
                for key in supervised_loss.keys()
            }

            # 提取总损失
            total_loss = loss_output['total_loss']

            # 梯度累积
            # 缩放后的损失值用于反向传播和梯度累积
            scaled_total_loss = total_loss / accumulation_steps
            scaled_total_loss.backward()    

            total_norm = 0
            '''gradient clip'''
            if (batch_id + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.7, norm_type=2)
                
                # 计算梯度范数
                total_norm = 0
                parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                # 教师模型的 EMA 更新 在每个 batch 的末尾更新教师模型权重
                # 每次调用 EMA 更新时，传入计算得到的 importance_weights
                ema_update(tea_net, model, 
                        step=batch_id+e*min_len,
                        total_steps=config['other']['nepoch']*min_len,
                        initial_momentum=initial_momentum,
                        importance_weights=importance_weights, eta=config['Stage2']['eta'])  # 这里 eta 可根据需要调整
                # ema_update(tea_net, model, step= batch_id+e*min_len, total_steps=config['other']['nepoch']*min_len,initial_momentum = config['Stage2']['ema_alpha'])     
                    # ema_update(tea_net, model, step= curren_step+e*total_ema_iter, total_steps=config['other']['nepoch']*total_ema_iter,initial_momentum = config['pseudo_label']['ema_alpha'])

                optimizer.step()
                optimizer.zero_grad()
            

            psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))
            msg = '{:0>8},[epoch {}] ({}/{}): total_loss = {}, rgb_loss = {}, eikonal_loss = {}, depth_loss = {}, normal_l1 = {}, normal_cos = {}, ray_mask_loss = {}, instance_mask_loss = {}, sdf_loss = {}, vis_sdf_loss = {}, psnr = {}, bete={}, alpha={} '.format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    e,
                    batch_id + 1,
                    min_len,
                    total_loss.item(),
                    loss_output['rgb_loss'].item(),
                    loss_output['eikonal_loss'].item(),
                    loss_output['depth_loss'].item(),
                    loss_output['normal_l1'].item(),
                    loss_output['normal_cos'].item(),
                    loss_output['ray_mask_loss'].item(),
                    loss_output['instance_mask_loss'].item(),
                    loss_output['sdf_loss'].item(),
                    loss_output['vis_sdf_loss'].item(),
                    psnr.item(),
                    model.module.density.get_beta().item(),
                    1. / model.module.density.get_beta().item(),
                    # model_outputs['depth_weight'].item()
                )
            cfg.log_string(msg)

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

            tb_logger.add_scalar('Statistics/beta', model.module.density.get_beta().item(), iter)
            tb_logger.add_scalar('Statistics/alpha', 1. / model.module.density.get_beta().item(), iter)
            tb_logger.add_scalar('Statistics/psnr', psnr.item(), iter)
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            tb_logger.add_scalar("train/lr", current_lr, iter)

            tb_logger.add_scalar('train/tea_loss', weight.item(), iter)
            tb_logger.add_scalar('train/sdf_var', sdf_var.item(), iter)
            tb_logger.add_scalar('train/pseudo_W', pseudo_W, iter)

            iter += 1

            # 在每个 epoch 结束后写入关键信息
            with open(weight_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([batch_id + 1, f"{weight.item():.6f}", f"{sdf_var.item():.6f}",
                                f"{pseudo_W:.6f}"])
        
        # 调整学习率
        scheduler.step()

        # after model_save_interval epoch, evaluate the model
        if e % config['other']['model_save_interval'] == 0:
            model.eval()
            eval_loss = 0
            eval_sdf_loss = 0
            tea_eval_loss = 0
            tea_eval_sdf_loss = 0
            StuTea = False


            eval_loss_info = {
            }

            tea_eval_loss_info = {
            }
            cfg.log_string("Switch Phase to Test")
            for batch_id, (indices, model_input, ground_truth) in enumerate(test_loader):
                torch.cuda.empty_cache()
            # 确保所有数据在 GPU 上
                model_input = {k: v.cuda().to(torch.float32) for k, v in model_input.items()}
                ground_truth = {k: v.cuda().to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in ground_truth.items()}
                model_outputs = model(model_input, indices)

                tea_model_outputs = tea_net(model_input,  indices)

                loss_output = loss(model_outputs, ground_truth, e)
                tea_loss_output = loss(tea_model_outputs, ground_truth, e)


                total_loss = loss_output['total_loss']
                tea_total_loss = tea_loss_output['total_loss']
            

                psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))
                tea_psnr = get_psnr(tea_model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))

                msg = 'Validation_Student {:0>8},[epoch {}] ({}/{}): total_loss = {}, rgb_loss = {}, eikonal_loss = {}, depth_loss = {}, normal_l1 = {}, normal_cos = {}, ray_mask_loss = {}, instance_mask_loss = {}, sdf_loss = {}, vis_sdf_loss = {}, psnr = {}, bete={}, alpha={}'.format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    e,
                    batch_id + 1,
                    len(test_loader),
                    total_loss.item(),
                    loss_output['rgb_loss'].item(),
                    loss_output['eikonal_loss'].item(),
                    loss_output['depth_loss'].item(),
                    loss_output['normal_l1'].item(),
                    loss_output['normal_cos'].item(),
                    loss_output['ray_mask_loss'].item(),
                    loss_output['instance_mask_loss'].item(),
                    loss_output['sdf_loss'].item(),
                    loss_output['vis_sdf_loss'].item(),
                    psnr.item(),
                    model.module.density.get_beta().item(),
                    1. / model.module.density.get_beta().item()
                )
                cfg.log_string(msg)

                msg = 'Validation_Teacher {:0>8},[epoch {}] ({}/{}): total_loss = {}, rgb_loss = {}, eikonal_loss = {}, depth_loss = {}, normal_l1 = {}, normal_cos = {}, ray_mask_loss = {}, instance_mask_loss = {}, sdf_loss = {}, vis_sdf_loss = {}, psnr = {}, bete={}, alpha={}'.format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    e,
                    batch_id + 1,
                    len(test_loader),
                    tea_total_loss.item(),
                    tea_loss_output['rgb_loss'].item(),
                    tea_loss_output['eikonal_loss'].item(),
                    tea_loss_output['depth_loss'].item(),
                    tea_loss_output['normal_l1'].item(),
                    tea_loss_output['normal_cos'].item(),
                    tea_loss_output['ray_mask_loss'].item(),
                    tea_loss_output['instance_mask_loss'].item(),
                    tea_loss_output['sdf_loss'].item(),
                    tea_loss_output['vis_sdf_loss'].item(),
                    tea_psnr.item(),
                    tea_net.module.density.get_beta().item(),
                    1. / tea_net.module.density.get_beta().item()
                )
                cfg.log_string(msg)

                for key in loss_output:
                    if "total" not in key:
                        if key not in eval_loss_info:
                            eval_loss_info[key] = 0
                        eval_loss_info[key] += torch.mean(loss_output[key]).item()

                for key in tea_loss_output:
                    if "total" not in key:
                        if key not in tea_eval_loss_info:
                            tea_eval_loss_info[key] = 0
                        tea_eval_loss_info[key] += torch.mean(tea_loss_output[key]).item()


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

            # 计算loss 用于决定是否保存模型                
            avg_eval_loss = eval_loss / (batch_id + 1)
            avg_sdf_loss = eval_sdf_loss / (batch_id + 1)
            tea_avg_eval_loss = tea_eval_loss / (batch_id + 1)
            tea_avg_sdf_loss = tea_eval_sdf_loss / (batch_id + 1)

            for key in eval_loss_info:
                eval_loss_info[key] = eval_loss_info[key] / (batch_id + 1)
            eval_loss_msg = f'avg_eval_loss is {avg_eval_loss}'
            eval_sdf_loss_msg = f'eval_sdf_loss_msg is {avg_sdf_loss}'
            cfg.log_string(eval_loss_msg)
            cfg.log_string(eval_sdf_loss_msg)
            tb_logger.add_scalar('eval/eval_loss', avg_eval_loss, e)
            for key in eval_loss_info:
                tb_logger.add_scalar("eval/" + key, eval_loss_info[key], e)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_eval_loss)


            for key in tea_eval_loss_info:
                tea_eval_loss_info[key] = tea_eval_loss_info[key] / (batch_id + 1)
            tea_eval_loss_msg = f'tea_avg_eval_loss is {tea_avg_eval_loss}'
            tea_eval_sdf_loss_msg = f'tea_eval_sdf_loss_msg is {tea_avg_sdf_loss}'
            cfg.log_string(tea_eval_loss_msg)
            cfg.log_string(tea_eval_sdf_loss_msg)
            tb_logger.add_scalar('eval/tea_eval_loss', tea_avg_eval_loss, e)
            for key in tea_eval_loss_info:
                tb_logger.add_scalar("eval/tea_" + key, tea_eval_loss_info[key], e)
                


            # 按照 epoch 数量动态创建文件夹来保存模型
            # 动态创建文件夹路径
            epoch_folder = os.path.join(log_dir, f'train_epoch_{e}')
            best_folder = os.path.join(log_dir, 'train_best')

            # 确保文件夹存在
            os.makedirs(epoch_folder, exist_ok=True)
            os.makedirs(best_folder, exist_ok=True)

            # 注册两个模型checkpoint
            checkpoint.register_modules(epoch=e, iter=iter, min_loss=avg_eval_loss)
            tea_checkpoint.register_modules(epoch=e, iter=iter, min_loss=tea_avg_eval_loss)


            # 保存最新的学生模型
            checkpoint.cfg.config['log']['path'] = epoch_folder
            checkpoint.save("latest_student")
            cfg.log_string(f"Student model saved to {epoch_folder}/latest_student.pth")
            # 保存最新的教师模型
            tea_checkpoint.cfg.config['log']['path'] = epoch_folder
            tea_checkpoint.save("latest_teacher")
            cfg.log_string(f"Teacher model saved to {epoch_folder}/latest_teacher.pth")

            # 如果是最佳学生模型
            if avg_eval_loss < min_eval_loss:
                checkpoint.cfg.config['log']['path'] = best_folder
                checkpoint.save('best_student')
                min_eval_loss = avg_eval_loss
                cfg.log_string(f"Best student model updated: {best_folder}/best_student.pth")

            # 如果是最佳教师模型
            if tea_avg_eval_loss < min_eval_loss_teacher:
                tea_checkpoint.cfg.config['log']['path'] = best_folder
                tea_checkpoint.save("best_teacher")
                min_eval_loss_teacher = tea_avg_eval_loss
                cfg.log_string(f"Best teacher model updated: {best_folder}/best_teacher.pth")

            # Meta-Controller adjust m_slow
            delta = torch.tensor(tea_avg_sdf_loss - avg_sdf_loss, device=device)
            epoch_norm = torch.tensor(e / config['other']['nepoch'], device=device)
            t_eval= torch.tensor(tea_avg_eval_loss,    device=device)
            gamma = controller(delta,t_eval,epoch_norm).item()
            # update m_slow with gamma
            initial_momentum = float(max(min(gamma * initial_momentum, 0.9996), 0.96))
            if e%3 == 0: 
                if e >= 70:
                    if  avg_eval_loss - tea_avg_eval_loss < 0.004:
                        ema_update(tea_net, model, 
                                step=e,
                                total_steps=config['other']['nepoch'],
                                initial_momentum=initial_momentum*0.6)  # 这里 eta 可根据需要调整
                        # tea_net.load_state_dict(model.state_dict())  # 将 tea_net 的参数复制到 model 中
                        StuTea = True


                else:
                    if avg_sdf_loss - tea_avg_sdf_loss < 0.002:
                        ema_update(tea_net, model, 
                                step=e,
                                total_steps=config['other']['nepoch'],
                                initial_momentum=initial_momentum*0.6)  # 这里 eta 可根据需要调整
                        StuTea = True
            
            # clean_increment_ema(tea_net, model, train_loader, initial_momentum, device,loss,e)       
            
            # 在每个 epoch 结束后写入关键信息
            with open(trend_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([e, f"{avg_eval_loss:.6f}", f"{avg_sdf_loss:.6f}",
                                f"{tea_avg_eval_loss:.6f}", f"{tea_avg_sdf_loss:.6f}",
                                f"{initial_momentum:.6f}", config['Stage2']['eta'], StuTea])

                 