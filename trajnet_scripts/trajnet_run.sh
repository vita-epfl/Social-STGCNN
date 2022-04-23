cd ..

MODES=3

# Train the model 
CUDA_VISIBLE_DEVICES=1 python3 trajnet_train.py \
    --dataset colfree_trajdata --obs_seq_len 9 --num_epochs 250 \
    --lr 0.01 --n_stgcnn 1 --n_txpcnn 5   \
    --tag social-stgcnn-trajnet-data --use_lrschd  \
    --fill_missing_obs 0 --keep_single_ped_scenes 0 --batch_size 32

# Evaluate on Trajnet++ 
CUDA_VISIBLE_DEVICES=1 python3 -m trajnet_evaluator \
    --dataset_name colfree_trajdata --write_only --modes ${MODES} \
    --fill_missing_obs 1 --keep_single_ped_scenes 1 --batch_size 1 
