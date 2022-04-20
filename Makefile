@ train_gpu:
	# CUDA_VISIBLE_DEVICES=1 python3 trajnet_train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset eth_data --tag social-stgcnn-eth-data --use_lrschd --num_epochs 250
	# CUDA_VISIBLE_DEVICES=1 python3 trajnet_train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset univ_data --tag social-stgcnn-univ-data --use_lrschd --num_epochs 250
	# CUDA_VISIBLE_DEVICES=1 python3 trajnet_train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara2_data --tag social-stgcnn-zara2-data --use_lrschd --num_epochs 250
	# CUDA_VISIBLE_DEVICES=1 python3 trajnet_train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset hotel_data --tag social-stgcnn-hotel-data --use_lrschd --num_epochs 250
	# CUDA_VISIBLE_DEVICES=1 python3 trajnet_train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara1_data --tag social-stgcnn-zara1-data --use_lrschd --num_epochs 250
	CUDA_VISIBLE_DEVICES=1 python3 trajnet_train.py \
		--lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset colfree_trajdata \
		--tag social-stgcnn-trajnet-data --use_lrschd --obs_seq_len 9 \
		--num_epochs 250 --fill_missing_obs 0 --keep_single_ped_scenes 0


@ test_gpu:
	# For ETH-UCY
	CUDA_VISIBLE_DEVICES=1 python3 trajnet_test.py
	# For TrajNet++
	# CUDA_VISIBLE_DEVICES=1 python3 trajnet_test.py --obs_seq_len 9

@ test_trajnet:
	CUDA_VISIBLE_DEVICES=1 python3 -m trajnet_evaluator \
		--dataset_name colfree_trajdata --write_only
		