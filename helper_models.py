import torch
from trajnet_loader import seq_to_graph


class DummyGCN:
    """
    Takes a single-pedestrian scene and makes static predictions.
    """
    def __init__(self, model, args):
        self.model = model
        self.obs_len = args.obs_len

    def __call__(self, obs_traj, obs_traj_rel, norm_lap_matr):
        """
        - pad obs_traj as in GAT
        - convert to V and A
        - predict
        - return only primary
        """
        
        # Initializing a static dummy pedestrian somewhere far away in the frame
        dummy_coords = torch.tensor([-1000., -1000.]).view(2, 1)
        zero_vels = torch.tensor([0., 0.]).view(2, 1)

        # Initialize with ones [peds, coords, frames]
        obs_traj_padded = torch.ones(2, 2, self.obs_len)
        obs_traj_rel_padded = torch.ones(2, 2, self.obs_len)

        # Keep the primary pedestrian
        obs_traj_padded[0, :, :] = obs_traj[0, :, :]
        obs_traj_rel_padded[0, :, :] = obs_traj_rel[0, :, :]

        # Add the static dummy pedestrian
        obs_traj_padded[1, :, :] = dummy_coords
        obs_traj_rel_padded[1, :, :] = zero_vels

        # Convert to graph
        V_obs_padded, A_obs_padded = \
            seq_to_graph(obs_traj_padded, obs_traj_rel_padded, norm_lap_matr)

        V_obs_padded = torch.unsqueeze(V_obs_padded, 0)
        V_obs_padded_tmp = V_obs_padded.permute(0, 3, 1, 2)

        # Compute model predictions
        V_obs_padded_tmp = V_obs_padded_tmp.cuda()
        A_obs_padded = A_obs_padded.cuda()
 
        V_pred_padded, _ = self.model(V_obs_padded_tmp, A_obs_padded.squeeze())

        # Return only the primary pedestrian
        return torch.unsqueeze(V_pred_padded[..., 0], 3)
