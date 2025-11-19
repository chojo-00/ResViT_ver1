cd "/workspace/nas100/forGPU/bc_cho/2_Code/ResViT"

# 입력한 뒤에


 python3 train.py \
 --dataroot "/workspace/mnt/nas206/ANO_DET/GAN_body/Pulmonary_Embolism/sampled_data/CCY_PE_DECT/journal_data/internal" \
 --name dect_enhancement_resvit \
 --gpu_ids 0 \
 --model resvit_one \
 --which_model_netG resvit \
 --dataset_mode dect_paired \
 --input_nc 1 \
 --output_nc 1 \
 --src 80keV,84keV,90keV,100keV,110keV,125keV \
 --trg 70keV \
 --lambda_A 100 \
 --norm batch \
 --pool_size 0 \
 --loadSize 512 \
 --fineSize 512 \
 --niter 25 \
 --niter_decay 25 \
 --save_epoch_freq 5 \
 --checkpoints_dir nas100/forGPU/bc_cho/2_Code/ResViT/checkpoints \
 --display_id 0 \
 --lr 0.001



 # 입력