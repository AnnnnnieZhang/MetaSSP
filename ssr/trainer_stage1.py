import os
import time
import datetime
import csv


import torch
from torch.utils.tensorboard import SummaryWriter

# from SSP3D.discriminatorPointsSDF import SDFDiscriminatorWithSurfacePoints

from utils.rend_util import get_psnr
from utils.sdf_util import get_sdf_gt_worldcoords
from SSP3D.feature_saver import FeatureSaver
def preprocess_loss(loss_value, c=0.4):
    """
    对输入的Loss值进行预处理
    Args:
        loss_value (float): 原始Loss值
        c (float): 控制值范围的参数

    Returns:
        float: 预处理后的Loss值
    """
    # 使用一个分段处理的非线性映射
    return loss_value / (loss_value + c)

def calculate_p_loss(loss_value, beta=1.0, alpha=0.001, k=6.0, gamma=30.0, m=3.0):
    """
    计算单个Loss值的P_loss
    Args:
        loss_value (float): 输入的单个Loss值
        beta (float): 控制最大值
        alpha (float): 平移参数，避免L=0时无穷大
        k (float): 控制反比例敏感度
        gamma (float): 指数衰减参数
        m (float): 加速指数衰减

    Returns:
        float: 计算得到的P_loss值
    """
    # 对输入进行预处理
    loss_value_processed = torch.tensor(preprocess_loss(loss_value), dtype=torch.float32)
    
    # 更新公式
    exp_term = torch.exp(-gamma * loss_value_processed**m)  # 指数衰减项
    frac_term = beta / (1 + (loss_value_processed + alpha) ** k)  # 改进的反比例项
    p_loss = exp_term * frac_term  # 结合两者
    
    # 确保范围在 [0, 1]
    p_loss = torch.clamp(p_loss, 0, 1)
    return p_loss.item()


def Recon_trainer(cfg,model,loss,optimizer,scheduler,train_loader,test_loader,device,checkpoint):
    start_t = time.time()
    config = cfg.config

    log_dir = cfg.save_path
    os.makedirs(log_dir, exist_ok=True)

    cfg.write_config()
    tb_logger = SummaryWriter(log_dir)

    # 定义保存 trend_stage1.csv 的路径
    trend_file_path = os.path.join(log_dir, 'trend_stage1.csv')

    # 确保文件存在（如果不存在则创建） 写入标题行
    if not os.path.exists(trend_file_path):
        with open(trend_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "AvgEvalLoss", "AvgSDFLoss", "train_batchsize", "train_lr"])
    start_epoch = 0
    iter = 0

    if config["resume"] == True:
        checkpoint.load(config["weight"])
        # start_epoch = scheduler.last_epoch
        start_epoch = checkpoint.module_dict['epoch']
        iter = checkpoint.module_dict['iter']
    if config['finetune']==True:
        start_epoch=0
    scheduler.last_epoch = start_epoch
    batchsize = config["data"]["batch_size"]["train"]*config["data"]["accumulation_steps"]
    lr = config["optimizer"]["lr"]

    
    # # 初始化判别器及优化器
    # discriminator = SDFDiscriminatorWithSurfacePoints(input_dim=4, hidden_dim=128, dropout_prob=0.1).to(device)
    # disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['pseudo_label']['discriminator_lr'])

    lambda_adv = config['pseudo_label']['lambda_adv']
    min_disc_loss = float('inf')  # 判别器最小损失

    # 检查是否需要加载预训练判别器权重
    # if config["dict_resume"] == True:
    #     # 拼接判别器权重文件路径
    #     discriminator_checkpoint_path = os.path.join(
    #         config["save_root_path"],
    #         config["exp_name"],
    #         config["dict_weight"]
    #     )

    #     # 尝试加载判别器权重
    #     if os.path.exists(discriminator_checkpoint_path):
    #         discriminator.load_state_dict(torch.load(discriminator_checkpoint_path, map_location=device))
    #         cfg.log_string(f"Discriminator model loaded from {discriminator_checkpoint_path}")
    #     else:
    #         raise FileNotFoundError(f"Discriminator checkpoint not found at {discriminator_checkpoint_path}")
    # else:
    #     cfg.log_string("Starting training with a fresh discriminator.")

    model.train()
    # discriminator.train()

    min_eval_loss = 10000
    
    accumulation_steps = config["data"]["accumulation_steps"]  # 累积步数，尝试不同的值
    get_protofile = config['pseudo_label']['get_protofile']
    
    for e in range(start_epoch, config['other']['nepoch']):
        if get_protofile:
            feature_saver = FeatureSaver(save_dir="SSP3D/proto_features")
        torch.cuda.empty_cache()

        cfg.log_string("Switch Phase to Train")
        model.train()
        # discriminator.train()

        optimizer.zero_grad()   # 重置优化器梯度
        # disc_optimizer.zero_grad()   # 重置优化器梯度

        for batch_id, (indices, model_input, ground_truth) in enumerate(train_loader):
            
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
            model_input = {k: v.cuda().to(torch.float32) for k, v in model_input.items()}
            ground_truth = {k: v.cuda().to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in ground_truth.items()}
            


            model_outputs = model(model_input, indices)
            loss_output = loss(model_outputs, ground_truth, e)



            # === 判别器训练 ===
            real_points = model_outputs['sample_points'].view(-1, 3)  # [batch_size, num_points, 3]
            real_sdf, obj_coords = get_sdf_gt_worldcoords(model_outputs['sample_points'], ground_truth, return_obj_coords = True)  # 真实 SDF [batch_size, num_points, 1]
            fake_sdf = model_outputs['sdf'].reshape(-1, 1)  # 伪造 SDF [batch_size, num_points, 1]

            # 拼接坐标和 SDF 值，作为判别器输入
            real_input = torch.cat([real_points, real_sdf], dim=-1)  # 拼接
            fake_input = torch.cat([real_points, fake_sdf], dim=-1)  # [batch_size, num_points, 4]

            if get_protofile:
                cls_indics = model_outputs['cls']
                batch_size = cls_indics.shape[0]  # 等于 15
                # 检查点数是否整除
                if real_sdf.shape[0] % batch_size != 0:
                    raise ValueError(f"real_sdf 的总点数 {real_sdf.shape[0]} 不能被 batch_size {batch_size} 整除")
                # 调整 real_sdf 的形状
                num_points = real_sdf.shape[0] // batch_size  # 等于 6272
                real_sdf_batch = real_sdf.view(batch_size, num_points, -1)  # [B, 6272, 1]
                obj_coords = obj_coords.view(batch_size, -1, 3)
                sdf_proto = torch.cat([obj_coords, real_sdf_batch], dim=-1)  # [B, 6272, 4]

                # 保存特征
                feature_saver.save_feature(cls_indics, sdf_proto)
            

            # # 随机丢弃部分特征
            # mask = torch.rand_like(real_input) > 0.5
            # real_input = real_input * mask
            # fake_input = fake_input * mask

            # 判别器输出
            # real_output = discriminator(real_input, batch_size = len(model_outputs['sample_points']))  # [batch_size, num_points, 1]
            # fake_output = discriminator(fake_input.detach(), batch_size = len(model_outputs['sample_points']))  # [batch_size, num_points, 1]

            # 使用目标扰动（Label Smoothing）：
            # real_targets = torch.ones_like(real_output) - 0.1 * torch.rand_like(real_output)
            # fake_targets = torch.zeros_like(fake_output) + 0.1 * torch.rand_like(fake_output)
            # real_loss = torch.nn.BCEWithLogitsLoss()(real_output, real_output)
            # fake_loss = torch.nn.BCEWithLogitsLoss()(fake_output, fake_output)

            # real_loss = torch.nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output))
            # fake_loss = torch.nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output))
            # adv_loss =  torch.nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output))

            # discriminator_loss = real_loss + fake_loss
            # scaled_discriminator_loss = discriminator_loss / (accumulation_steps/2)
            # scaled_discriminator_loss.backward() 


            total_loss = loss_output['total_loss']
            # 加上对抗损失
            # total_loss = total_loss + lambda_adv * adv_loss.detach()

            # 梯度累积
            # 缩放后的损失值用于反向传播和梯度累积
            scaled_total_loss = total_loss / accumulation_steps
            scaled_total_loss.backward()    

            # total_loss.backward()

            # scaled_discriminator_loss = discriminator_loss / accumulation_steps
            # scaled_discriminator_loss.backward()

            total_norm = 0
            '''gradient clip'''
            # if (batch_id + 1) % (accumulation_steps/2) == 0:
            #     torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.7, norm_type=2)
            #     # 优化判别器
            #     disc_optimizer.step() 
            #     disc_optimizer.zero_grad()

            if (batch_id + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.7, norm_type=2)
                
                # 计算梯度范数
                total_norm = 0
                parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                optimizer.step()
                optimizer.zero_grad()            

                # # 优化判别器
                # disc_optimizer.step() 
                # disc_optimizer.zero_grad()
                           

            psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))
            msg = '{:0>8},[epoch {}] ({}/{}): total_loss = {}, rgb_loss = {}, eikonal_loss = {}, depth_loss = {}, normal_l1 = {}, normal_cos = {}, ray_mask_loss = {}, instance_mask_loss = {}, sdf_loss = {}, vis_sdf_loss = {}, psnr = {}, bete={}, alpha={}'.format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    e,
                    batch_id + 1,
                    len(train_loader),
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
                    # model_outputs['depth_weight'].item()
                    # discriminator_loss.item()
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

            # tb_logger.add_scalar('Statistics/dino_weight', model_outputs['dino_weight'].item(), iter)
            tb_logger.add_scalar('Statistics/beta', model.module.density.get_beta().item(), iter)
            tb_logger.add_scalar('Statistics/alpha', 1. / model.module.density.get_beta().item(), iter)
            tb_logger.add_scalar('Statistics/psnr', psnr.item(), iter)
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            tb_logger.add_scalar("train/lr", current_lr, iter)

            iter += 1
        if get_protofile:
            feature_saver.write_to_file()

        # 调整学习率
        scheduler.step()

        # after model_save_interval epoch, evaluate the model
        if e % config['other']['model_save_interval'] == 0:
            model.eval()
            # discriminator.eval()
            
            eval_loss = 0
            eval_sdf_loss = 0
            # total_disc_loss = 0
            eval_loss_info = {
            }
            cfg.log_string("Switch Phase to Test")
            for batch_id, (indices, model_input, ground_truth) in enumerate(test_loader):
                torch.cuda.empty_cache()
                model_input = {k: v.cuda().to(torch.float32) for k, v in model_input.items()}
                ground_truth = {k: v.cuda().to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in ground_truth.items()}

                model_outputs = model(model_input, indices)
                loss_output = loss(model_outputs, ground_truth, e)

                # if ground_truth['rgb'].shape[0] < batch_size:
                #     print(f"Skipping last batch: insufficient data ({ground_truth['rgb'].shape[0]} < {batch_size})")
                #     continue

                # # === 判别器训练 ===
                with torch.no_grad():
                #     real_points = model_outputs['sample_points'].view(-1, 3)  # [batch_size, num_points, 3]

                #     real_sdf = get_sdf_gt_worldcoords(model_outputs['sample_points'], ground_truth)  # 真实 SDF
                #     fake_sdf = model_outputs['sdf'].reshape(-1, 1)

                #     # 拼接坐标和 SDF 值，作为判别器输入
                #     real_input = torch.cat([real_points, real_sdf], dim=-1)  # 拼接
                #     fake_input = torch.cat([real_points, fake_sdf], dim=-1)  # [batch_size, num_points, 4]
                    

                    # # 判别器输出
                    # real_output = discriminator(real_input, batch_size = len(model_outputs['sample_points']))  # [batch_size, num_points, 1]
                    # fake_output = discriminator(fake_input.detach(), batch_size = len(model_outputs['sample_points']))  # [batch_size, num_points, 1]

                    # real_loss = torch.nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output))
                    # fake_loss = torch.nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output))

                    # disc_loss = real_loss + fake_loss
                    # total_disc_loss += disc_loss.item()
                    
                    
                    total_loss = loss_output['total_loss']

                    # total_loss = total_loss + lambda_adv * fake_loss.detach()
            

                psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))
                msg = 'Validation {:0>8},[epoch {}] ({}/{}): total_loss = {}, rgb_loss = {}, eikonal_loss = {}, depth_loss = {}, normal_l1 = {}, normal_cos = {}, ray_mask_loss = {}, instance_mask_loss = {}, sdf_loss = {}, vis_sdf_loss = {}, psnr = {}, bete={}, alpha={}'.format(
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
                    # disc_loss.item()
                )
                cfg.log_string(msg)

                for key in loss_output:
                    if "total" not in key:
                        if key not in eval_loss_info:
                            eval_loss_info[key] = 0
                        eval_loss_info[key] += torch.mean(loss_output[key]).item()

                if torch.isnan(total_loss).item():
                    print("NaN detected in total_loss!")
                else:
                    eval_loss += total_loss.item()
                    eval_sdf_loss += loss_output['sdf_loss'].item()

            # 计算平均  loss
            # avg_disc_loss = total_disc_loss / (batch_id + 1)
            avg_eval_loss = eval_loss / (batch_id + 1)
            avg_sdf_loss = eval_sdf_loss / (batch_id + 1)

            for key in eval_loss_info:
                eval_loss_info[key] = eval_loss_info[key] / (batch_id + 1)
            eval_loss_msg = f'avg_eval_loss is {avg_eval_loss}'
            # dict_loss_msg = f'avg_dict_loss is {avg_disc_loss}'

            cfg.log_string(eval_loss_msg)
            # cfg.log_string(dict_loss_msg)

            tb_logger.add_scalar('eval/eval_loss', avg_eval_loss, e)
            # tb_logger.add_scalar('eval/dict_loss', avg_disc_loss, e)

            for key in eval_loss_info:
                tb_logger.add_scalar("eval/" + key, eval_loss_info[key], e)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_eval_loss)
            # else:
            #     scheduler.step()

            checkpoint.register_modules(epoch=e, iter=iter, min_loss=avg_eval_loss)
            if avg_eval_loss < min_eval_loss:
                checkpoint.save('best')
                min_eval_loss = avg_eval_loss
            else:
                checkpoint.save("latest")

            # # 判别器模型保存逻辑
            # if avg_disc_loss < min_disc_loss:
            #     min_disc_loss = avg_disc_loss
            #     torch.save(discriminator.state_dict(), os.path.join(log_dir, 'disc_best.pth'))
            # else:
            #     torch.save(discriminator.state_dict(), os.path.join(log_dir, 'disc_latest.pth'))

            # 在每个 epoch 结束后写入关键信息
            with open(trend_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([e, f"{avg_eval_loss:.6f}", f"{avg_sdf_loss:.6f}",
                                f"{batchsize}", f"{lr}"])
                cfg.log_string(f"Trend information saved to {trend_file_path}")



