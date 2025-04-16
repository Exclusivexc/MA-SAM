for i in 41 68 137 206 275; do
    for j in 19 39 59 79 99 119 139 159 179 199 219 239 259 279 299 319 339 359 379; do
        CUDA_VISIBLE_DEVICES=1 python ../test.py \
            --adapt_ckpt "/media/ubuntu/maxiaochuan/MA-SAM/outputs/my_select_${i}_b20/epoch_${j}.pth" \
            --data_path "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/2D_all_5slice" \
            --ckpt "/media/ubuntu/maxiaochuan/MA-SAM/sam_vit_b_01ec64.pth" --stage "test" 
    done
done