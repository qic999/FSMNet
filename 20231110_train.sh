#!/bin/bash
pip install einops h5py matplotlib scikit_image tensorboardX yacs pandas opencv-python timm ml_collections -i https://pypi.tuna.tsinghua.edu.cn/simple
cd /data/qic99/recon_code/BRATS_freq5

### without guidance modality, compare the image domain performance with under_mri and kspace_recon as input.
# nohup python train_SOTA.py --root_path /data/qic99/MRI_recon/image_100patients_4X/ \
#     --gpu 4 --model_name unet_single --batch_size 4 --base_lr 0.0001 --MRIDOWN 4X --low_field_SNR 0 \
#     --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
#     --exp unet_wo_kspace_lr1e-4 > "logs/$(date +"%Y%m%d-%H%M")_unet_4X_baseline.log" 2>&1 &

# python train_SOTA.py --root_path /data/qic99/MRI_recon/image_100patients_4X/ \
#     --gpu 0 --model_name unet_single --batch_size 4 --base_lr 0.0002 --MRIDOWN 4X --low_field_SNR 0 \
#     --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
#     --exp unet_wo_kspace_4X_nosigmoid > "logs/$(date +"%Y%m%d-%H%M")_unet_4X_baseline.log" 2>&1 &
python train_SOTA.py --root_path /data/qic99/MRI_recon/image_100patients_4X/ \
    --gpu 0 --model_name unet_single --batch_size 4 --base_lr 0.0001 --MRIDOWN 4X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp debug

python train_SOTA.py --root_path /data/qic99/MRI_recon/image_100patients_4X/ \
    --gpu 0 --model_name unet_single --batch_size 4 --base_lr 0.0001 --MRIDOWN 4X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp unet_wo_kspace_4X_lr1e-4

python train_SOTA.py --root_path /data/qic99/MRI_recon/image_100patients_8X/ \
    --gpu 1 --model_name unet_single --batch_size 4 --base_lr 0.0001 --MRIDOWN 8X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp unet_wo_kspace_8X_lr1e-4

python train_fastmri.py --root_path /data/qic99/MRI_recon/fastMRI/ \
    --gpu 0 --model_name unet_single --batch_size 4 --base_lr 0.0001 --CENTER_FRACTIONS 0.08 --ACCELERATIONS 4 \
    --exp our_fastmri_4x

python train_fastmri.py --root_path /data/qic99/MRI_recon/fastMRI/ \
    --gpu 1 --model_name unet_single --batch_size 4 --base_lr 0.0001 --CENTER_FRACTIONS 0.04 --ACCELERATIONS 8 \
    --exp our_fastmri_8x

python train_fastmri.py --root_path /data/qic99/MRI_recon/fastMRI/ \
    --gpu 0 --model_name unet_single --batch_size 4 --base_lr 0.0005 --CENTER_FRACTIONS 0.08 --ACCELERATIONS 4 \
    --exp our_fastmri_4x_lr0.0005

python train_fastmri.py --root_path /data/qic99/MRI_recon/fastMRI/ \
    --gpu 1 --model_name unet_single --batch_size 4 --base_lr 0.0005 --CENTER_FRACTIONS 0.04 --ACCELERATIONS 8 \
    --exp our_fastmri_8x_lr0.0005

python train_fastmri.py --root_path /data/qic99/MRI_recon/fastMRI/ \
    --gpu 0 --model_name unet_single --batch_size 4 --base_lr 0.001 --CENTER_FRACTIONS 0.08 --ACCELERATIONS 4 \
    --exp our_fastmri_4x_lr0.001

python train_fastmri.py --root_path /data/qic99/MRI_recon/fastMRI/ \
    --gpu 1 --model_name unet_single --batch_size 4 --base_lr 0.001 --CENTER_FRACTIONS 0.04 --ACCELERATIONS 8 \
    --exp our_fastmri_8x_lr0.001

python test_SOTA.py --root_path /data/qic99/MRI_recon/image_100patients_4X/ \
    --gpu 0 --model_name unet_single --base_lr 0.0001 --MRIDOWN 4X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp unet_wo_kspace_4X_lr1e-4 --phase test

python visualize_erf.py --root_path /data/qic99/MRI_recon/image_100patients_4X/ \
    --gpu 0 --model_name unet_single --batch_size 4 --base_lr 0.0001 --MRIDOWN 4X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp test --resume model/unet_wo_kspace_4X_lr1e-4/best_checkpoint.pth

python analyze_erf.py --source visualize.npy --heatmap_save heatmap.png
pip install matplotlib==3.3 -i https://pypi.tuna.tsinghua.edu.cn/simple


python visualize_erf.py --root_path /data/qic99/MRI_recon/image_100patients_4X/ \
    --gpu 0 --model_name unet_single --batch_size 4 --base_lr 0.0001 --MRIDOWN 4X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp test --resume model/unet_wo_kspace_4X/best_checkpoint.pth

python test_SOTA.py --root_path /data/qic99/MRI_recon/image_100patients_4X/ \
    --gpu 3 --model_name unet_single --base_lr 0.0001 --MRIDOWN 4X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp unet_wo_kspace_4X_lr1e-4 --phase test

python test_SOTA.py --root_path /data/qic99/MRI_recon/image_100patients_8X/ \
    --gpu 4 --model_name unet_single --base_lr 0.0001 --MRIDOWN 8X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp unet_wo_kspace_8X_lr1e-4 --phase test

python test_SOTA.py --root_path /data/qic99/MRI_recon/image_100patients_4X/ \
    --gpu 3 --model_name unet_single --base_lr 0.0001 --MRIDOWN 4X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp unet_wo_kspace_4X_lr1e-4 --phase test

python test_SOTA.py --root_path /data/qic99/MRI_recon/image_100patients_8X/ \
    --gpu 4 --model_name unet_single --base_lr 0.0001 --MRIDOWN 8X --low_field_SNR 0 \
    --kspace_refine False --use_multi_modal False --modality t2 --input_normalize mean_std \
    --exp unet_wo_kspace_8X_lr1e-4 --phase test

python test_fastmri.py --root_path /data/qic99/MRI_recon/fastMRI/ \
    --gpu 5 --model_name unet_single --batch_size 4 --base_lr 0.0001 --CENTER_FRACTIONS 0.08 --ACCELERATIONS 4 \
    --exp our_fastmri_4x --phase test

python test_fastmri.py --root_path /data/qic99/MRI_recon/fastMRI/ \
    --gpu 6 --model_name unet_single --batch_size 4 --base_lr 0.0001 --CENTER_FRACTIONS 0.04 --ACCELERATIONS 8 \
    --exp our_fastmri_8x --phase test
    