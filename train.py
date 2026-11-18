"""MetaSSP training entry point.

The stage is selected by ``Stage`` in the config file:
  Stage 1 - supervised warm-up on the labeled subset      (configs/pix3d_warmup.yaml)
  Stage 2 - semi-supervised teacher-student training      (configs/pix3d_semi.yaml)

Usage:
    python train.py --config configs/pix3d_warmup.yaml
    python train.py --config configs/pix3d_semi.yaml
"""
import os
import argparse

import cv2
import torch
import torch.nn as nn

from configs.config_utils import CONFIG
from ssr.ssr_utils.utils import load_device, get_model, get_loss, \
    get_dataloader, CheckpointIO, get_trainer, get_optimizer, load_scheduler
from ssr.ssr_utils.network_utils import fix_random_seed

# avoid Qt plugin conflicts between opencv and other packages
dirname = os.path.dirname(cv2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


def parse_args():
    parser = argparse.ArgumentParser('MetaSSP training')
    parser.add_argument('--config', type=str, required=True, help='configure file for training.')
    return parser.parse_args()


def run(cfg):
    torch.set_default_dtype(torch.float32)
    stage = cfg.config.get('Stage', 1)
    semi = stage == 2

    if cfg.config['fix_random_seed']:
        fix_random_seed(seed=1029)

    cfg.log_string('Data save path: %s' % (cfg.save_path))
    checkpoint = CheckpointIO(cfg)

    cfg.log_string('Loading device settings.')
    device = load_device(cfg)

    cfg.log_string('Loading dataset.')
    train_loader = get_dataloader(cfg.config, mode='train')
    test_loader = get_dataloader(cfg.config, mode='val')
    if semi:
        pseudo_train_loader = get_dataloader(cfg.config, mode='train', pseudo=True)

    cfg.log_string('Setting loss function')
    loss = get_loss(cfg.config, mode='train')

    cfg.log_string('Loading model.')
    net = get_model(cfg.config, device=device).float()

    if semi:
        # The teacher is initialised from the warm-up (Stage 1) checkpoint.
        cfg.log_string('Loading teacher model.')
        tea_checkpoint = CheckpointIO(cfg)
        tea_net = nn.DataParallel(get_model(cfg.config, device=device).float()).to(device)
        tea_checkpoint.register_modules(net=tea_net)
        tea_checkpoint.load(cfg.config['Stage2']['teacher_weight'])

    cfg.log_string('Loading optimizer.')
    optimizer = get_optimizer(config=cfg.config, net=net)
    checkpoint.register_modules(opt=optimizer)
    net = nn.DataParallel(net).to(device)
    checkpoint.register_modules(net=net)

    cfg.log_string('Loading optimizer scheduler.')
    scheduler = load_scheduler(config=cfg.config, optimizer=optimizer, train_loader=train_loader)
    checkpoint.register_modules(sch=scheduler)

    cfg.log_string('Loading trainer.')
    trainer = get_trainer(cfg.config)

    cfg.log_string('Start to train.')
    if semi:
        trainer(cfg, net, tea_net, loss, optimizer, scheduler, train_loader=train_loader,
                pseudo_train_loader=pseudo_train_loader, test_loader=test_loader, device=device,
                checkpoint=checkpoint, tea_checkpoint=tea_checkpoint)
    else:
        trainer(cfg, net, loss, optimizer, scheduler, train_loader=train_loader,
                test_loader=test_loader, device=device, checkpoint=checkpoint)

    cfg.log_string('Training finished.')


if __name__ == "__main__":
    args = parse_args()
    cfg = CONFIG(args.config)
    cfg.update_config(args.__dict__)

    cfg.log_string('Loading configuration')
    cfg.log_string(cfg.config)
    cfg.write_config()

    cfg.log_string('Training begin.')
    run(cfg)
