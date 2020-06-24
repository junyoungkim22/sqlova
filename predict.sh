CUDA_VISIBLE_DEVICES=$1 python predict.py --bert_path=bert_and_wikisql --bert_type_abb uS --split=dev --data_path=$2

