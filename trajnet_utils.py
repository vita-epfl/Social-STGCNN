import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time

import trajnetplusplustools

def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)

def drop_distant(xy, r=5.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask]
       
def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr: 
            G = nx.from_numpy_matrix(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def pre_process_test(sc_, obs_len=8):
    obs_frames = [primary_row.frame for primary_row in sc_[0]][:obs_len]
    last_frame = obs_frames[-1]
    sc_ = [[row for row in ped] for ped in sc_ if ped[0].frame <= last_frame]
    return sc_


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, dataset, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True, test=False):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.test = test

        seq_list = []
        seq_list_rel = []
        num_peds_in_seq = []
        for _, (filename, scene_id, paths) in enumerate(dataset):
            if self.test:
                paths = pre_process_test(paths, self.obs_len)
            ## Get new scene
            pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)
            pos_scene = drop_distant(pos_scene)
            # Removing Partial Tracks. Model cannot account for it !! NaNs in Loss
            full_traj = np.isfinite(pos_scene).all(axis=2).all(axis=0)
            seq = pos_scene[:, full_traj]
            # Make Rel Scene
            seq_rel = np.zeros_like(seq)
            seq_rel[1:] = seq[1:] - seq[:-1]
            
            if sum(full_traj) > min_ped:
                seq_list.append(seq)
                seq_list_rel.append(seq_rel)
                num_peds_in_seq.append(sum(full_traj))

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=1).transpose(1, 2, 0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=1).transpose(1, 2, 0)
        self.dummy = torch.from_numpy(np.array([0.0])).type(torch.float)
        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.v_pred = [] 
        self.A_pred = [] 
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]
            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.dummy, self.dummy,
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        return out
