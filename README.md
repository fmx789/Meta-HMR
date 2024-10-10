# Meta-HMR

Offcial implementation of  "Incorporating Test-Time Optimization into Training with Dual Networks for Human Mesh Recovery"

## Getting Started

This code was tested on `Ubuntu 18.04.6 LTS` and requires:

* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU

### Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:

```
git clone https://github.com/fmx789/Meta-HMR
cd Meta-HMR

# Conda environment setup
conda env create -f requirements.yml
conda activate metahmr
```

### Data Preparation

1. Download [the SMPL models](https://smpl.is.tue.mpg.de/) for rendering the reconstructed meshes.
2. Download raw images for each dataset (Human 3.6M, 3DPW, MPI-INF-3DHP, MPII and COCO 2014) and necessary SMPL files from official websites respectively. Pseudo-GT for COCO & MPII can be downloaded from [[CLIFF](https://drive.google.com/drive/folders/1EmSZwaDULhT9m1VvH7YOpCXwBWgYrgwP?usp=sharing)], and other files can be downloaded from [[SPIN](https://github.com/nkolot/SPIN)].
3. Download the pretrained weights for ResNet-50 and HRNet-w48 from [[Google Drive](https://drive.google.com/drive/folders/1-0Fq0wGwzFuQ-iB-WKkHMoz0AEz-Jycl?usp=sharing)].
4. Download the pretrained checkpoints from [[Google Drive 1](https://drive.google.com/drive/folders/1-0Fq0wGwzFuQ-iB-WKkHMoz0AEz-Jycl?usp=sharing)] and [[Google Drive 2](https://drive.google.com/drive/folders/1EmSZwaDULhT9m1VvH7YOpCXwBWgYrgwP?usp=sharing)].
5. Place the data in the following directory structure:

```
${ROOT}
|-- ckpt
	|-- cliff_hrnet_3dpw.pt
	|-- cliff_hrnet_h36m.pt
	|-- hmr_resnet_3dpw.pt
	|-- hmr_resnet_h36m.pt
|-- data
    |-- smpl_mean_params.npz
    |-- J_regressor_h36m.npy
    |-- J_regressor_extra.npy
    |-- datasets 
    	# Path for annotations; please modify the names as necessary
    	|-- 3dpw_train.npz
    	|-- 3dpw_test.npz
    	|-- ...
    |-- VOC
    	|-- pascal_occluders.npy
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_MALE.pkl
        |-- SMPL_NEUTRAL.pkl
|-- models
	|-- backbones
		|-- pose_hrnet_w48_256x192.pth
		|-- pose_resnet.pth
	|-- ckpt 
		# Checkpoints for auxiliary network initialization
		|-- hmr.pt
		|-- res50-PA45.7_MJE72.0_MVE85.3_3dpw.pt
		|-- hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt
```

6. Modify the file paths in [config.py](./config.py) according to your environment settings.

## Train

To train the model using `CLIFF`, run:

```
python train.py --name <train model name> --batch_size 30 --test_epochs 5 --bbox_type rect --inner_lr 1e-5 --outer_lr 1e-4 --after_innerlr 1e-5 --inner_step 1 --test_val_step 14 --first_order --backbone resnet --adapt_val_step --model cliff --syn_occ --num_epochs 30 --summary_steps 1000
```

To train the model using `HMR`, set the parameter `--model hmr`.

To use the `HRNet` backbone, set the parameter `--backbone hrnet`.

To evaluate performance using 2D keypoints detected by OpenPose, add the parameter `--use_openpose2d`.

To finetune on a pretrained checkpoint, add the parameter `--pretrained_checkpoint <path to previous checkpoint>`.

You can decide whether or not to use the `3DPW` training set by modifying the [relevant lines](https://github.com/fmx789/Meta-HMR/blob/main/train/trainer_one_stage_cliff.py#L35-L36) in the code.

For more information about parameters, please refer to the instructions in [train_options.py](./utils/train_options.py).

## Evaluation

To evaluate `CLIFF` model on `3DPW`, run:

```
python eval_cliff.py --dataset 3dpw --bbox_type rect --backbone hrnet --batch_size 20 --inner_lr 1e-5 --after_innerlr 1e-5 --inner_step 1 --test_val_step 14 --adapt_val_step --checkpoint <path to 3DPW checkpoint>
```

To evaluate `HMR` model on `3DPW`, run:

```
python eval_hmr.py --dataset 3dpw --bbox_type rect --backbone resnet --batch_size 20 --inner_lr 1e-5 --after_innerlr 1e-5 --inner_step 1 --test_val_step 14 --adapt_val_step --checkpoint <path to 3DPW checkpoint>
```

To evaluate performance using OpenPose detected 2D keypoints, add the parameter `--use_openpose2d`.

To test on `H36M`, set the parameter `--dataset h36m-p2` and change the parameter `--checkpoint` to the H36M checkpoint.

To view additional visualization results, add the parameter `--viz`.

## Citation

If you find this repo useful, please consider citing:

```
@inproceedings{metahmr,
title = {Incorporating Test-Time Optimization into Training with Dual Networks for Human Mesh Recovery},
author = {Nie, Yongwei and Fan, Mingxian and Long, Chengjiang and Zhang, Qing and Zhu, Jian and Xu, Xuemiao},
booktitle={Proceedings of the thirty-eighth Annual Conference on Neural Information Processing Systems},
year = {2024}
}
```

## Acknowledgement

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public.

- [SPIN](https://github.com/nkolot/SPIN)
- [PARE](https://github.com/mkocabas/PARE)
- [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)
- [ReFit](https://github.com/yufu-wang/ReFit)
- [PyMAF](https://github.com/HongwenZhang/PyMAF)
