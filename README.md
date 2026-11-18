# MetaSSP

Official code for **"MetaSSP: Enhancing Semi-supervised Implicit 3D Reconstruction through Meta-adaptive EMA and SDF-aware Pseudo-label Evaluation"**
Luoxi Zhang, Chun Xie, Itaru Kitahara (University of Tsukuba) — [arXiv:2602.06163](https://arxiv.org/abs/2602.06163)

MetaSSP is a semi-supervised framework for SDF-based single-view 3D reconstruction. Starting from a
supervised warm-up on 10% of the labels, a teacher–student pipeline exploits the remaining unlabeled
images through:

- **Gradient-based parameter importance** — every epoch, the teacher's sensitivity to each parameter is
  estimated on unlabeled data from the gradient of `||sdf||²`.
- **Meta-adaptive EMA** — the teacher is updated with a cosine-annealed momentum that is rescaled by a
  small meta-controller, and each parameter's update is damped by its importance.
- **SDF-aware pseudo-label weighting** — pseudo-labels are weighted by the teacher's consistency under
  weak/strong augmentation and by the variance of its predicted SDF.

On Pix3D with 10% labels, the paper reports a Chamfer Distance of 3.12 (×10²) and a mean IoU of 36.37%.

## Repository layout

```
MetaSSP
├── train.py                  # training entry point (stage selected by the config)
├── inference.py              # mesh extraction for evaluation / visualisation
├── configs
│   ├── pix3d_warmup.yaml     # Stage 1: supervised warm-up (10% labeled)
│   └── pix3d_semi.yaml       # Stage 2: semi-supervised training (10% labeled + 90% unlabeled)
├── metassp                   # the semi-supervised components of the paper
│   ├── importance.py         #   gradient-based parameter importance   (Sec. III-C)
│   ├── pseudo_label.py       #   pseudo-label weight and loss blending  (Eq. 4-5)
│   ├── ema.py                #   meta-adaptive, importance-regularised EMA (Sec. III-D)
│   └── augmentation.py       #   weak / strong augmentations            (Sec. III-E)
├── ssr                       # SSR backbone: model, losses, dataloaders, trainers
│   ├── trainer.py            #   Stage 1 trainer
│   └── trainer_semi.py       #   Stage 2 trainer (teacher-student loop)
├── tools
│   ├── split_pix3d.py        # build the labeled / unlabeled splits
│   └── convert_ssmp_split.py # convert the SSMP split file to this repo's format
├── eval                      # Chamfer Distance / F-score / normal consistency
├── utils                     # rendering, SDF and plotting helpers
└── external                  # third-party code: gaps (ICP alignment), Chamfer distance
```

## Installation

The experiments in the paper were run with Python 3.10, PyTorch 2.4.1 and CUDA 12.4 on a single GPU.

```bash
conda create -n metassp python=3.10
conda activate metassp
# install the PyTorch build that matches your CUDA version, e.g.
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`environment.yml` is the environment of the original SSR code base (Python 3.8 / PyTorch 1.10) and is kept
for reference.

Evaluation additionally needs [gaps](https://github.com/tomfunkhouser/gaps) for ICP alignment:

```bash
cd external
bash build_gaps.sh
```

## Data preparation

MetaSSP uses the preprocessed Pix3D data and the split files of
[SSR](https://github.com/DaLi-Jack/SSR-code).

1. Download the preprocessed data from the SSR repository and unzip it into `data/`.
2. Copy the `dataset/` folder (split files) of the SSR repository into the root of this repository.
3. Build the labeled / unlabeled training splits:

   ```bash
   python tools/split_pix3d.py --percent 10
   ```

   For every category this takes the first 10% of `dataset/Pix3D/split/train/all.json` as the labeled
   subset and writes `10%.json` and `90%.json` next to it.

Expected layout:

```
MetaSSP
├── data
│   └── Pix3D                 # images, depth, normals, gt-sdf, ...
└── dataset
    └── Pix3D
        └── split
            ├── train
            │   ├── all.json
            │   ├── 10%.json  # labeled   (data.train_class_name)
            │   └── 90%.json  # unlabeled (data.pseudo_class_name)
            ├── val
            │   └── all.json
            └── test
                └── all.json
```

## Training

Training has two stages; both are launched with `train.py` and differ only in the config.

**Stage 1 — supervised warm-up** on the labeled 10%:

```bash
python train.py --config configs/pix3d_warmup.yaml
```

The best checkpoint is saved to `output/pix3d_warmup/model_best.pth`.

**Stage 2 — semi-supervised training.** Teacher and student are initialised from the warm-up checkpoint
given by `Stage2.teacher_weight`:

```bash
python train.py --config configs/pix3d_semi.yaml
```

Outputs are written to `output/pix3d_semi/`:

| File | Content |
| --- | --- |
| `train_best/model_best_teacher.pth` | best teacher (the model used for evaluation) |
| `train_best/model_best_student.pth` | best student |
| `train_epoch_<e>/model_latest_{teacher,student}.pth` | checkpoints of epoch `e` |
| `trend_stage2.csv` | per-epoch validation losses of student and teacher, EMA momentum, γ, reset flag |
| `weight_stage2.csv` | per-batch consistency loss, SDF variance and pseudo-label weight |
| `log.txt`, TensorBoard events | full training log |

Monitor training with `tensorboard --logdir output`.

### Settings

| | Warm-up | Semi-supervised |
| --- | --- | --- |
| Data | 10% labeled Pix3D | 10% labeled + 90% unlabeled Pix3D |
| Epochs | 400 | 200 |
| Effective batch size | 128 | 160 |
| LR decay epochs | [350, 390] | [170, 190] |

The effective batch size is `data.accumulation_steps × (batch_size.train + batch_size.pseudo)`; the
configs use 16 accumulation steps with a per-step batch of 8 (warm-up) and 2 labeled + 8 unlabeled
(semi-supervised). Lower the per-step batch and raise `accumulation_steps` if you run out of GPU memory.
Use `device.gpu_ids` (e.g. `'0,1'`) for multi-GPU training.

The semi-supervised hyper-parameters are in the `Stage2` block of `configs/pix3d_semi.yaml`:

| Key | Default | Meaning |
| --- | --- | --- |
| `ema_alpha` | 0.996 | initial EMA momentum m₀ of the cosine schedule |
| `ema_min`, `ema_max` | 0.96, 0.9996 | clamp of the final momentum `m = clip(γ · m_base)` |
| `eta` | 0.06 | strength of the importance regularisation η |
| `importance_batches` | 100 | unlabeled batches per epoch used to estimate importance (N) |
| `pseudo_alpha`, `pseudo_beta` | 4, 4 | penalties on consistency loss and SDF variance (α, β) |
| `pseudo_lambda` | 0.2 | maximum contribution of the pseudo-label loss (λ) |
| `reset_*` | — | when and how the low-momentum "reset" EMA update is applied |

## Inference and evaluation

Set `exp_name` and `weight` in the config to the experiment and checkpoint to evaluate
(`configs/pix3d_semi.yaml` points to the best teacher by default), keep `eval.export_mesh: True` and
`eval.export_color_mesh: False`, then run:

```bash
bash eval/evaluate.sh configs/pix3d_semi.yaml
```

This runs `inference.py` (meshes are written to `output/<exp_name>/out/`), then computes Chamfer
Distance and F-score (`eval/evaluate_front3d.py`) and normal consistency
(`eval/evaluate_normal_consistency.py`). Each step can also be run on its own with `--config`.

To export textured meshes for visualisation, set `eval.export_color_mesh: True` and run
`python inference.py --config configs/pix3d_semi.yaml`.

## Notes

- The 3D-FRONT / SUN RGB-D configs, dataloaders and `inference_rot_angle.py` / `inference_sunrgbd.py`
  come from SSR and are kept for supervised training; the semi-supervised stage is implemented for Pix3D.

## Citation

```bibtex
@article{zhang2026metassp,
  title   = {MetaSSP: Enhancing Semi-supervised Implicit 3D Reconstruction through Meta-adaptive EMA and SDF-aware Pseudo-label Evaluation},
  author  = {Zhang, Luoxi and Xie, Chun and Kitahara, Itaru},
  journal = {arXiv preprint arXiv:2602.06163},
  year    = {2026}
}
```

## Acknowledgements

This code is built on [SSR](https://github.com/DaLi-Jack/SSR-code) (Chen et al., "Single-view 3D Scene
Reconstruction with High-fidelity Shape and Texture"), which in turn borrows from InstPIFu, PixelNeRF and
MonoSDF. We thank the authors for releasing their code.
