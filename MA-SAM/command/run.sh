CUDA_VISIBLE_DEVICES=2 python ../train.py --root_path "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/2D_slices" \
    --output "/media/ubuntu/maxiaochuan/MA-SAM/outputs/UNSAM_u10_Embed_3" --ckpt "/media/ubuntu/maxiaochuan/MA-SAM/sam_vit_b_01ec64.pth" --batch_size "20" \
    --csv_path "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/2D_slices/csv_files/training.csv" \
    --finetune_method "MA_UN_SAM" \
    --vit_name "vit_b_Un"