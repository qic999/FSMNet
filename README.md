# FSMNet



## Paper

<b>Accelerated Multi-Contrast MRI Reconstruction via Frequency and Spatial Mutual Learning</b> <br/>
[Qi Chen](https://scholar.google.com/citations?user=4Q5gs2MAAAAJ&hl=en)<sup>1</sup>, [Xiaohan Xing](https://hathawayxxh.github.io/)<sup>2, *</sup>, [Zhen Chen](https://franciszchen.github.io/)<sup>3</sup>, [Zhiwei Xiong](http://staff.ustc.edu.cn/~zwxiong/)<sup>1</sup> <br/>
<sup>1 </sup>University of Science and Technology of China,  <br/>
<sup>2 </sup>Stanford University,  <br/>
<sup>3 </sup>Centre for Artificial Intelligence and Robotics (CAIR), HKISI-CAS  <br/>
MICCAI, 2024 <br/>
[paper]() | [code](https://github.com/qic999/FSMNet) | [huggingface]()

## 0. Installation

```bash
git clone https://github.com/qic999/FSMNet.git
cd FSMNet
```

See [installation instructions](documents/INSTALL.md) to create an environment and obtain requirements.

## 1. Training
##### BraTS dataset, AF=4
```
python train_brats.py --root_path /data/qic99/MRI_recon image_100patients_4X/ \
    --gpu 0 --model_name unet_single --batch_size 4 --base_lr 0.0001 --MRIDOWN 4X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp FSMNet_BraTS_4x
```

##### BraTS dataset, AF=8
```
python train_brats.py --root_path /data/qic99/MRI_recon/image_100patients_8X/ \
    --gpu 1 --model_name unet_single --batch_size 4 --base_lr 0.0001 --MRIDOWN 8X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp FSMNet_BraTS_8x
```

##### fastMRI dataset, AF=4
```
python train_fastmri.py --root_path /data/qic99/MRI_recon/fastMRI/ \
    --gpu 0 --model_name unet_single --batch_size 4 --base_lr 0.0001 --CENTER_FRACTIONS 0.08 --ACCELERATIONS 4 \
    --exp FSMNet_fastmri_4x
```

##### fastMRI dataset, AF=8
```
python train_fastmri.py --root_path /data/qic99/MRI_recon/fastMRI/ \
    --gpu 1 --model_name unet_single --batch_size 4 --base_lr 0.0001 --CENTER_FRACTIONS 0.04 --ACCELERATIONS 8 \
    --exp FSMNet_fastmri_8x
```

## 2. Testing
##### BraTS dataset, AF=4
```
python test_brats.py --root_path /data/qic99/MRI_recon/image_100patients_4X/ \
    --gpu 3 --model_name unet_single --base_lr 0.0001 --MRIDOWN 4X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp FSMNet_BraTS_4x --phase test
```

##### BraTS dataset, AF=8
```
python test_brats.py --root_path /data/qic99/MRI_recon/image_100patients_8X/ \
    --gpu 4 --model_name unet_single --base_lr 0.0001 --MRIDOWN 8X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp FSMNet_BraTS_8x --phase test
```

##### fastMRI dataset, AF=4
```
python test_fastmri.py --root_path /data/qic99/MRI_recon/fastMRI/ \
    --gpu 5 --model_name unet_single --batch_size 4 --base_lr 0.0001 --CENTER_FRACTIONS 0.08 --ACCELERATIONS 4 \
    --exp FSMNet_fastmri_4x --phase test
```

##### fastMRI dataset, AF=8
```
python test_fastmri.py --root_path /data/qic99/MRI_recon/fastMRI/ \
    --gpu 6 --model_name unet_single --batch_size 4 --base_lr 0.0001 --CENTER_FRACTIONS 0.04 --ACCELERATIONS 8 \
    --exp FSMNet_fastmri_8x --phase test
```