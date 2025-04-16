CUDA_VISIBLE_DEVICES=0 python train.py --root_path "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/Fore_slices" \
    --output "/media/ubuntu/maxiaochuan/MA-SAM/outputs/foreslices_batch30" \
    --ckpt "/media/ubuntu/maxiaochuan/MA-SAM/sam_vit_b_01ec64.pth" \
    --batch_size 30