CUDA_VISIBLE_DEVICES=1 python ../test.py \
    --adapt_ckpt "/media/ubuntu/maxiaochuan/MA-SAM/outputs/UN_SAM_all_b20_u10/epoch_${a}.pth" \
    --data_path "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/2D_slices" \
    --finetune_method "MA_UN_SAM" \
    --vit_name "vit_b_Un" \
    --ckpt "/media/ubuntu/maxiaochuan/MA-SAM/sam_vit_b_01ec64.pth" --stage "test" 