@ train_gpu:
	CUDA_VISIBLE_DEVICES=1 python3 trajnet_train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset eth_data --tag social-stgcnn-eth-data --use_lrschd --num_epochs 250
	CUDA_VISIBLE_DEVICES=1 python3 trajnet_train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset univ_data --tag social-stgcnn-univ-data --use_lrschd --num_epochs 250
	CUDA_VISIBLE_DEVICES=1 python3 trajnet_train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara2_data --tag social-stgcnn-zara2-data --use_lrschd --num_epochs 250

@ test_gpu:
	CUDA_VISIBLE_DEVICES=1 python3 trajnet_test.py
