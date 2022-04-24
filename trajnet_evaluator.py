import os
import argparse
import pickle

from joblib import Parallel, delayed
import scipy
import torch
from tqdm import tqdm
import trajnetplusplustools
import numpy as np
import copy 

from evaluator.trajnet_evaluator import trajnet_evaluate
from evaluator.write_utils import \
    load_test_datasets, preprocess_test, write_predictions

from trajnet_loader import trajnet_loader
from helper_models import DummyGCN


# STGCNN
from model import social_stgcnn
import torch.distributions.multivariate_normal as torchdist
from metrics import seq_to_nodes, nodes_rel_to_nodes_abs


def predict_scene(model, batch, args):
    assert len(batch) == 10
    batch = [tensor.cuda() for tensor in batch]
    batch = [
        torch.unsqueeze(tensor, 0) if len(tensor.shape) == 3 else tensor \
        for tensor in batch
        ]
    
    obs_traj, pred_traj_gt, \
    obs_traj_rel, pred_traj_gt_rel, \
    non_linear_ped, loss_mask, \
    V_obs, A_obs, \
    V_tr, A_tr = \
        batch

    # Forward
    V_obs_tmp = V_obs.permute(0, 3, 1, 2)

    # If there's only one pedestrian, handle it differently
    if V_obs_tmp.shape[3] == 1:
        single_ped_model = DummyGCN(model, args)
        V_pred = single_ped_model(
            obs_traj[0, ...], obs_traj_rel[0, ...], args.norm_lap_matr
            )
    else: 
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
    V_pred = V_pred.permute(0, 2, 3, 1)

    # Remove the batch dimension
    V_tr = V_tr[0, ...]
    A_tr = A_tr[0, ...]
    V_pred = V_pred[0, ...]
    num_of_objs = obs_traj_rel.shape[1]

    V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:,:num_of_objs,:]
    
    # Fit the multivariate distribution
    sx = torch.exp(V_pred[:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,4]) #corr
    
    cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
    cov[:,:,0,0]= sx*sx
    cov[:,:,0,1]= corr*sx*sy
    cov[:,:,1,0]= corr*sx*sy
    cov[:,:,1,1]= sy*sy
    mean = V_pred[:,:,0:2]
    
    mvnormal = torchdist.MultivariateNormal(mean, cov)      

    # Rel to abs
    V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
    V_x_rel_to_abs = nodes_rel_to_nodes_abs(
        V_obs.data.cpu().numpy().copy()[0, ...], V_x[0,:,:].copy()
        )

    V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
    V_y_rel_to_abs = nodes_rel_to_nodes_abs(
        V_tr.data.cpu().numpy().copy(), V_x[-1,:,:].copy()
        )

    # Get the predictions and save them
    multimodal_outputs = {}
    for num_p in range(args.modes):
        
        # Sample a prediction
        V_pred = mvnormal.sample()

        V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
            V_pred.data.cpu().numpy().copy(), V_x[-1, :, :].copy()
            )
    
        output_primary = V_pred_rel_to_abs[:, 0]
        output_neighs = V_pred_rel_to_abs[:, 1:]
        multimodal_outputs[num_p] = [output_primary, output_neighs]

    return multimodal_outputs



def load_predictor(args):

    model = social_stgcnn(
        n_stgcnn=args.n_stgcnn,
        n_txpcnn=args.n_txpcnn,
        output_feat=args.output_size,
        seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,
        pred_seq_len=args.pred_seq_len
        ).cuda()

    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    return model


def get_predictions(args):
    """
    Get model predictions for each test scene and write the predictions 
    in appropriate folders.
    """
    # List of .json file inside the args.path 
    # (waiting to be predicted by the testing model)
    datasets = sorted([
        f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) \
        if not f.startswith('.') and f.endswith('.ndjson')
        ])

    # Extract Model names from arguments and create its own folder 
    # in 'test_pred' for storing predictions
    # WARNING: If Model predictions already exist from previous run, 
    # this process SKIPS WRITING
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')
        model_name = model_name + '_modes' + str(args.modes)

        ## Check if model predictions already exist
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        if not os.path.exists(args.path + model_name):
            os.makedirs(args.path + model_name)
        else:
            print(f'Predictions corresponding to {model_name} already exist.')
            print('Loading the saved predictions')
            continue

        print("Model Name: ", model_name)
        model = load_predictor(args)
        goal_flag = False

        # Iterate over test datasets
        for dataset in datasets:
            # Load dataset
            dataset_name, scenes, scene_goals = \
                load_test_datasets(dataset, goal_flag, args)

            # Convert it to a trajnet loader
            scenes_loader = trajnet_loader(
                scenes, 
                args, 
                drop_distant_ped=False, 
                test=True,
                keep_single_ped_scenes=args.keep_single_ped_scenes,
                fill_missing_obs=args.fill_missing_obs
                ) 

            # Can be removed; it was useful for debugging
            scenes_loader = list(scenes_loader)

            # Get all predictions in parallel. Faster!
            scenes_loader = tqdm(scenes_loader)
            pred_list = Parallel(n_jobs=args.n_jobs)(
                delayed(predict_scene)(model, batch, args)
                for batch in scenes_loader
                )
            
            # Write all predictions
            write_predictions(pred_list, scenes, model_name, dataset_name, args)


def main():
    # Define new arguments to overwrite the existing ones
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill_missing_obs", default=1, type=int)
    parser.add_argument("--keep_single_ped_scenes", default=1, type=int)
    parser.add_argument("--norm_lap_matr", default=1, type=int)
    parser.add_argument("--n_jobs", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--dataset_name", default="eth_data", type=str)
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default="./checkpoint/social-stgcnn-trajnet-data"
        )
    parser.add_argument(
        '--modes', default=1, type=int, help='number of modes to predict'
        )
    parser.add_argument(
        '--write_only', action='store_true', help='disable writing new files'
        )
    parser.add_argument(
        '--disable-collision', action='store_true', 
        help='disable collision metrics'
        )
    parser.add_argument(
        '--labels', required=False, nargs='+', help='labels of models'
        )
    parser.add_argument(
        '--normalize_scene', action='store_true', help='augment scenes'
        )

    new_args = parser.parse_args()

    # Load arguments that were used for training the particular checkpoint
    args_path = os.path.join(new_args.checkpoint_dir, 'args.pkl')
    with open(args_path, 'rb') as f: 
        args = pickle.load(f)

    # Overwrite certain fields
    args.fill_missing_obs = new_args.fill_missing_obs
    args.keep_single_ped_scenes = new_args.keep_single_ped_scenes
    args.norm_lap_matr = new_args.norm_lap_matr
    args.modes = new_args.modes
    args.n_jobs = new_args.n_jobs
    args.dataset_name = new_args.dataset_name
    args.write_only = new_args.write_only
    args.disable_collision = new_args.disable_collision
    args.labels = new_args.labels
    args.normalize_scene = new_args.normalize_scene
    args.batch_size = new_args.batch_size
    
    # Load corresponding statistics
    stats_path = os.path.join(new_args.checkpoint_dir, 'constant_metrics.pkl')
    with open(stats_path, 'rb') as f: 
        cm = pickle.load(f)
    print("Stats:", cm)

    # Add checkpoint paths
    args.checkpoint = os.path.join(new_args.checkpoint_dir, 'val_best.pth')
    args.path = os.path.join('datasets', args.dataset_name, 'test_pred/')
    args.output = [args.checkpoint]

    # Adding arguments with names that fit the evaluator module
    # in order to keep it unchanged
    args.obs_length = args.obs_seq_len
    args.pred_length = args.pred_seq_len
    
    # Writes to Test_pred
    # Does NOT overwrite existing predictions if they already exist ###
    get_predictions(args)
    if args.write_only: # For submission to AICrowd.
        print("Predictions written in test_pred folder")
        exit()

    ## Evaluate using TrajNet++ evaluator
    trajnet_evaluate(args)


if __name__ == '__main__':
    main()


