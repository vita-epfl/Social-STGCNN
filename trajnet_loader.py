import numpy as np
import torch
import math
import networkx as nx
from tqdm import tqdm

import trajnetplusplustools


def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)


def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
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


def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask]



def get_limits_of_missing_intervals(finite_frame_inds, obs_len):
    """
    Given a SORTED array of indices of finite frames per pedestrian, get the 
    indices which represent limits of NaN (missing) intervals in the array.
    Example (for one pedestrian):
        array = [3, 4, 5, 8, 9, 10, 13, 14, 15, 18]
        obs_len = 18

        ==>> result = [0, 3, 5, 8, 10, 13, 15, 18]
    The resulting array is an array with an even number of elements,
    because it represents pairs of start-end indices (i.e. limits) for 
    intervals that should be padded. 
        ==>> intervals to be padded later: [0, 3], [5, 8], [10, 13], [15, 18]
    """
    # Adding start and end indices
    if 0 not in finite_frame_inds:
        finite_frame_inds = np.insert(finite_frame_inds, 0, -1) 
    if obs_len not in finite_frame_inds:
        finite_frame_inds = \
            np.insert(finite_frame_inds, len(finite_frame_inds), obs_len)

    # Keeping only starts and ends of continuous intervals
    limits, interval_len = [], 1
    for i in range(1, len(finite_frame_inds)):
        # If this element isn't the immediate successor of the previous
        if finite_frame_inds[i] > finite_frame_inds[i - 1] + 1:
            if interval_len:
                # Add the end of the previous interval
                if finite_frame_inds[i - 1] == -1:
                    limits.append(0)
                else:
                    limits.append(finite_frame_inds[i - 1])
                # Add the start of the new interval
                limits.append(finite_frame_inds[i])
                # If this is a lone finite element, add the next interval
                if interval_len == 1 and i != len(finite_frame_inds) - 1 \
                    and finite_frame_inds[i + 1] > finite_frame_inds[i] + 1:
                    limits.append(finite_frame_inds[i])
                    limits.append(finite_frame_inds[i + 1])
            interval_len = 0
        else:
            interval_len += 1
            
    return limits


def fill_missing_observations(pos_scene_raw, obs_len, test):
    """
    Performs the following:
        - discards pedestrians that are completely absent in 0 -> obs_len
        - discards pedestrians that have any NaNs after obs_len
        - In 0 -> obs_len:
            - finds FIRST non-NaN and fill the entries to its LEFT with it
            - finds LAST non-NaN and fill the entries to its RIGHT with it
    """

    # Discarding pedestrians that are completely absent in 0 -> obs_len
    peds_are_present_in_obs = \
        np.isfinite(pos_scene_raw).all(axis=2)[:obs_len, :].any(axis=0)
    pos_scene = pos_scene_raw[:, peds_are_present_in_obs, :]

    if not test:
        # Discarding pedestrians that have NaNs after obs_len
        peds_are_absent_after_obs = \
            np.isfinite(pos_scene).all(axis=2)[obs_len:, :].all(axis=0)
        pos_scene = pos_scene[:, peds_are_absent_after_obs, :]

    # Finding indices of finite frames per pedestrian
    finite_frame_inds, finite_ped_inds = \
        np.where(np.isfinite(pos_scene[:obs_len]).all(axis=2))
    finite_frame_inds, finite_ped_inds = \
        finite_frame_inds[np.argsort(finite_ped_inds)], np.sort(finite_ped_inds)

    finite_frame_inds_per_ped = np.split(
        finite_frame_inds, np.unique(finite_ped_inds, return_index=True)[1]
        )[1:]
    finite_frame_inds_per_ped = \
        [np.sort(frames) for frames in finite_frame_inds_per_ped]

    # Filling missing frames
    for ped_ind in range(len(finite_frame_inds_per_ped)):
        curr_finite_frame_inds = finite_frame_inds_per_ped[ped_ind]

        # limits_of_cont_ints: [start_1, end_1, start_2, end_2, ... ]
        limits_of_missing_ints = \
            get_limits_of_missing_intervals(curr_finite_frame_inds, obs_len)
        assert len(limits_of_missing_ints) % 2 == 0
            
        i = 0
        while i < len(limits_of_missing_ints):
            start_ind, end_ind = \
                limits_of_missing_ints[i], limits_of_missing_ints[i + 1]
            # If it's the beginning (i.e. first element is NaN):
            #   - pad with the right limit, else use left
            #   - include start_ind, else exclude it
            if start_ind == 0 and not np.isfinite(pos_scene[0, ped_ind]).all():
                padding_ind = end_ind 
                start_ind = start_ind 
            else:
                padding_ind = start_ind
                start_ind = start_ind + 1

            pos_scene[start_ind:end_ind, ped_ind] = pos_scene[padding_ind, ped_ind]
            i += 2

    return pos_scene


def trajnet_loader(
    data_loader, 
    args, 
    drop_distant_ped=False, 
    test=False, 
    keep_single_ped_scenes=False,
    fill_missing_obs=False,
    norm_lap_matr=True
    ):
    """
    Will work only for batch_size = 1 (as it originally did as well).
    => seq_list, seq_list_rel, num_peds_in_seq - will have at the most 1 element
    """
    seq_len = args.obs_len + args.pred_len
    seq_list, seq_list_rel, num_peds_in_seq = [], [], []
    for batch_idx, (filename, scene_id, paths) in enumerate(data_loader):
        if test:
            paths = pre_process_test(paths, args.obs_len)

        ## Get new scene
        pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)
        if drop_distant_ped:
            pos_scene = drop_distant(pos_scene)

        ## Account for incomplete tracks (filling or discarding)
        if fill_missing_obs:
            seq = fill_missing_observations(pos_scene, args.obs_len, test)
            full_traj = np.isfinite(seq).all(axis=2).all(axis=0)
        else:
            # Removing Partial Tracks. Model cannot account for it !! NaNs in Loss
            full_traj = np.isfinite(pos_scene).all(axis=2).all(axis=0)
            seq = pos_scene[:, full_traj]
        
        # Make Rel Scene
        seq_rel = np.zeros_like(seq)
        seq_rel[1:] = seq[1:] - seq[:-1]

        # Discarding single pedestrian scenes (or keeping them)
        if not (sum(full_traj) > 1 or keep_single_ped_scenes):
            continue 
        
        # Adding the current scene to the list
        seq_list.append(seq)
        seq_list_rel.append(seq_rel)
        num_peds_in_seq.append(sum(full_traj))

        ##############################
        # Continue if not whole batch
        # (when/if adding that option)
        ##############################

        # Convert lists to torch arrays
        seq_list = np.concatenate(seq_list, axis=1).transpose(1, 2, 0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=1).transpose(1, 2, 0)
        
        # Convert numpy -> Torch Tensor
        dummy = torch.from_numpy(np.array([0.0])).type(torch.float)
        obs_traj = torch.from_numpy(seq_list[:, :, :args.obs_len])\
            .type(torch.float)
        pred_traj = torch.from_numpy(seq_list[:, :, args.obs_len:])\
            .type(torch.float)
        obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :args.obs_len])\
            .type(torch.float)
        pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, args.obs_len:])\
            .type(torch.float)
        
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
            ]

        # Make sure there's only one element
        assert len(seq_start_end) == 1

        # Convert current batch to nx graphs 
        v_obs, A_obs, v_pred, A_pred = [], [], [], []
        for ss in range(len(seq_start_end)):
            start, end = seq_start_end[ss]

            # Observations
            v_, a_ = seq_to_graph(
                obs_traj[start:end,:], obs_traj_rel[start:end, :], 
                norm_lap_matr
                )
            v_obs.append(v_.clone())
            A_obs.append(a_.clone())

            # Predictions
            v_, a_=seq_to_graph(
                pred_traj[start:end,:], pred_traj_rel[start:end, :], 
                norm_lap_matr
                )
            v_pred.append(v_.clone())
            A_pred.append(a_.clone())

        # Yield the only element 
        index = 0
        start, end = seq_start_end[index]
        out = [
            obs_traj[start:end, :], pred_traj[start:end, :],
            obs_traj_rel[start:end, :], pred_traj_rel[start:end, :],
            dummy, dummy,
            v_obs[index], A_obs[index],
            v_pred[index], A_pred[index]
            ]

        yield out

        seq_list, seq_list_rel, num_peds_in_seq = [], [], []

        
