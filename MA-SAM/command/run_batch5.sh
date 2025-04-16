CUDA_VISIBLE_DEVICES=1 python train.py --root_path "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/2D_all_5slice" \
    --output "/media/ubuntu/maxiaochuan/MA-SAM/outputs/batch6" --ckpt "/media/ubuntu/maxiaochuan/MA-SAM/sam_vit_b_01ec64.pth" --batch_size 6 \
    --csv_path "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/2D_all_5slice.csv"