CUDA_VISIBLE_DEVICES=1 python ../test.py \
    --adapt_ckpt "/media/ubuntu/maxiaochuan/MA-SAM/outputs/my_select_27_b20_seed2023/epoch_399.pth" \
    --data_path "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/2D_slices" \
    --ckpt "/media/ubuntu/maxiaochuan/MA-SAM/sam_vit_b_01ec64.pth" --stage "test" 