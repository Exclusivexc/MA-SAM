CUDA_VISIBLE_DEVICES=2 python ../test.py \
    --adapt_ckpt "/media/ubuntu/maxiaochuan/MA-SAM/outputs/my_select_13_b10/epoch_399.pth" \
    --data_path "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/2D_all_5slice" \
    --ckpt "/media/ubuntu/maxiaochuan/MA-SAM/sam_vit_b_01ec64.pth" --stage "test" 