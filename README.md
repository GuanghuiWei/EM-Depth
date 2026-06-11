# EM-Depth

**Enhanced Modeling for Self-Supervised Monocular Depth Estimation  [Paper link](https://ieeexplore.ieee.org/document/11303883/)**

Accepted by IEEE Transactions on Consumer Electronics (TCE), 2026

</a>  

<img src="./img/teaser_m.gif" width="100%" alt="teaser" align=center />

(EM-Depth*+NimbleD)

</div>

## Results 
| Method | Data Augmentation | Resolution| Train | Train Images | AbsRel | δ < 1.25<sup>1</sup> |
|  :----------:   | :--------:| :--------: |:--:| :--------: | :----: |:----: |
| [EM-Depth](https://drive.google.com/drive/folders/1vqofkpdZ2L0fSvmaFpkDN-KuGPpZpjcd?usp=sharing)     | SCM     | 640×192    | M  |  39,810    | 0.095 | 0.905 |
| [EM-Depth*](https://drive.google.com/drive/folders/1vqofkpdZ2L0fSvmaFpkDN-KuGPpZpjcd?usp=sharing)    | SCM+RC  | 640×192    | M  |  39,810    | 0.094 | 0.906 |
| [EM-Depth\*+NimbleD](https://drive.google.com/drive/folders/1vqofkpdZ2L0fSvmaFpkDN-KuGPpZpjcd?usp=sharing) | SCM+RC  | 640×192    | M+ |  39,810    | 0.089 | 0.914 |
| [EM-Depth](https://drive.google.com/drive/folders/1vqofkpdZ2L0fSvmaFpkDN-KuGPpZpjcd?usp=sharing)     | SCM     | 640×192    | M* |  71,634    | 0.079 | 0.927 |
| EM-Depth     | SCM     | 640×192    | S  |  39,810    | 0.095 | 0.889 |
| EM-Depth     | SCM     | 640×192    | MS |  39,810    | 0.093 | 0.908 |
| EM-Depth     | SCM     | 1024×320   | M  |  39,810    | 0.090 | 0.913 |
| [EM-Depth\*+NimbleD](https://drive.google.com/drive/folders/1vqofkpdZ2L0fSvmaFpkDN-KuGPpZpjcd?usp=sharing) | SCM+RC  | 1024×320   | M+ |  39,810    | 0.086 | 0.921 |
| EM-Depth     | SCM     | 1024×320   | S  |  39,810    | 0.090 | 0.899 |
|[EM-Depth\*+NimbleD](https://drive.google.com/drive/folders/1vqofkpdZ2L0fSvmaFpkDN-KuGPpZpjcd?usp=sharing) | SCM+RC  | 1280×384   | S+ |  45,200    | 0.084 | 0.913 |
| EM-Depth     | SCM     | 1024×320   | MS |  39,810    | 0.087 | 0.919 |
| EM-Depth     | SCM     | 1024×320   | MS*|  71,634    | 0.072 | 0.931 |


## Datasets 

### KITTI
Please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) to prepare  KITTI data.

### Cityscapes
Please refer to [DynamicDepth](https://github.com/AutoAILab/DynamicDepth) to prepare Cityscapes data.

### nuScenes
Please refer to [Dynamo-Depth](https://github.com/YihongSun/Dynamo-Depth) to prepare nuScenes data.

For nuScenes, we use a day-clear subset, and the corresponding file list is provided in `splits/nuScenes/train_files.txt`.


## Setup

Install the dependencies with:
```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install Albumentations==0.5.0 numpy==1.24.4 opencv-python==4.9.0.80 opencv-python-headless==4.9.0.80
pip install scikit-image scipy matplotlib pillow scipy timm thop
pip install linear-warmup-cosine-annealing-warm-restarts-weight-decay==1.0 # refer Lite-Mono
```

## Evaluation


```shell
python evaluate_depth.py \
--kitti_path /path/to/your_kitti \
--cityscapes_path /path/to/your_cityscapes \
--nuscenes_path /path/to/your_nuscenes \
--make3d_path /path/to/your_make3d \
--nyuv2_path /path/to/your_nyuv2 \
--height 192 \
--width 640 \
--load_weights_folder /path/to/your_weights \
# --use_stereo  # if use stereo
```

Specify the path of the dataset to be evaluated. For example, to evaluate only on KITTI, provide only:

```shell
python evaluate_depth.py \
--kitti_path /path/to/your_kitti \
--height 192 \
--width 640 \
--load_weights_folder /path/to/your_weights
```

## ⏳Training

### Large-scale video pre-training
For large-scale pretraining, please refer to NimbleD. We also provide our pretrained weights [DHRNet+NimbleD](https://drive.google.com/drive/folders/1vqofkpdZ2L0fSvmaFpkDN-KuGPpZpjcd?usp=sharing).

### Prepare pseudo-labels

```shell
python generate_kitti_pseudo_labels.py --data_dir /path/to/your_kitti # refer NimbleD
```

###  Start training

```shell
python train.py  --model_name mytrain --num_epochs 20 --batch_size 12 --lr 0.0001 5e-6 11 0.0001 1e-5 11 --scales 0
```

### Pre-training Weights
If you want to reproduce the results of EM-Depth, you can download the pretrained HRNet18 weights from the  [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification) repository.


## Acknowledgement

We have benefited from code released by many excellent open-source projects, including [Monodepth2](https://github.com/nianticlabs/monodepth2), [Lite-Mono](https://github.com/noahzn/Lite-Mono), [MonoViT](https://github.com/zxcqlf/monovit), [Dynamo-Depth](https://github.com/YihongSun/Dynamo-Depth), [RA-Depth](https://github.com/hmhemu/RA-Depth), [BDE-Depth](https://github.com/LiuJF1226/BDEdepth), [NimbleD](https://github.com/xapaxca/nimbled), and [Depth Anything](https://github.com/LiheYoung/Depth-Anything). We sincerely thank the authors for their excellent work and contributions to the community.

## Citation

```shell
@ARTICLE{11303883,
  author={Wei, Guanghui and Wang, Laihua and Li, Shuai and Gao, Xingyu and Qi, Sumin and Yu, Yang and Wu, Liehao},
  journal={IEEE Transactions on Consumer Electronics}, 
  title={Enhanced Modeling for Self-Supervised Monocular Depth Estimation}, 
  year={2026},
  volume={72},
  number={1},
  pages={682-691},
```
