
python train.py --gpu_ids 6 --seed 0 --loadSize 299 --fineSize 299 \
    --name gp3-faceforensics-nt --save_epoch_freq 200 \
    --real_im_path /data2/lp/IVBridge_datasets/crop/ \
    --fake_im_path /data2/lp/IVBridge_datasets/crop/ \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience} \
	--which_model_netD xception_block5 --model patch_discriminator \
	--patience 10 --lr_policy constant  --max_epochs 1000 \
    --overwrite_config