CUDA_VISIBLE_DEVICES=$1 python3 train.py --accumulate_gradients 2 --lr 0.001 --lr_bert 0.00001 --max_seq_leng 222 \
    --do_train --bS 8 --fine_tune  --bert_type_abb uS --seed $2 --input=$3/$2
