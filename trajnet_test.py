import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import * 
from metrics import * 
from model import social_stgcnn
import copy

## TrajNet++
import trajnetplusplustools
from data_load_utils import prepare_data
from trajnet_utils import TrajectoryDataset
from trajnetpp_eval_utils import trajnet_sample_eval, trajnet_sample_multi_eval

def test(KSTEPS=20):
    global loader_test,model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step =0

    ade_tot, fde_tot, pred_col_tot, gt_col_tot = 0., 0., 0., 0.
    topk_ade_tot, topk_fde_tot = 0., 0.
    num_batch = 0
    for batch in loader_test: 
        num_batch += 1
        step+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch


        num_of_objs = obs_traj_rel.shape[1]

        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())
        # print(V_pred.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])
        V_pred = V_pred.permute(0,2,3,1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()


        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]
        #print(V_pred.shape)

        #For now I have my bi-variate parameters 
        #normx =  V_pred[:,:,0:1]
        #normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr
        
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda()
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]
        
        mvnormal = torchdist.MultivariateNormal(mean,cov)


        ### Rel to abs 
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 
        
        #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                 V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                 V_x[-1,:,:].copy())
        
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        # for n in range(num_of_objs):
        #     ade_ls[n]=[]
        #     fde_ls[n]=[]

        multi_preds = []
        for k in range(KSTEPS):

            V_pred = mvnormal.sample()



            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            
            multi_preds.append(copy.deepcopy(V_pred_rel_to_abs).transpose(1, 0, 2))
            if k == 0:
                s_ade, s_fde, s_pred_col, s_gt_col = trajnet_sample_eval(V_pred_rel_to_abs.transpose(1, 0, 2), 
                                                                         V_y_rel_to_abs.transpose(1, 0, 2))
                ade_tot += s_ade
                fde_tot += s_fde
                pred_col_tot += s_pred_col
                gt_col_tot += s_gt_col

        s_topk_ade, s_topk_fde = trajnet_sample_multi_eval(multi_preds, V_y_rel_to_abs.transpose(1, 0, 2))
        topk_ade_tot += s_topk_ade
        topk_fde_tot += s_topk_fde

    print(num_batch)
    ade_tot /= num_batch
    fde_tot /= num_batch
    pred_col_tot /= num_batch
    gt_col_tot /= num_batch
    topk_ade_tot /= num_batch
    topk_fde_tot /= num_batch
    return ade_tot,fde_tot, pred_col_tot, gt_col_tot, topk_ade_tot, topk_fde_tot


paths = ['./checkpoint/*social-stgcnn*']
KSTEPS=20

print("*"*50)
print('Number of samples:',KSTEPS)
print("*"*50)




for feta in range(len(paths)):
    ade_ls = [] 
    fde_ls = [] 
    path = paths[feta]
    exps = glob.glob(path)
    print('Model being tested are:',exps)

    for exp_path in exps:
        print("*"*50)
        print("Evaluating model:",exp_path)

        model_path = exp_path+'/val_best.pth'
        args_path = exp_path+'/args.pkl'
        with open(args_path,'rb') as f: 
            args = pickle.load(f)

        stats= exp_path+'/constant_metrics.pkl'
        with open(stats,'rb') as f: 
            cm = pickle.load(f)
        print("Stats:",cm)



        #Data prep     
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        test_dataset, _, _ = prepare_data('datasets/' + args.dataset, subset='/test_private/', sample=1.0)
        dset_test = TrajectoryDataset(
                test_dataset,
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1,norm_lap_matr=True,
                test=True)

        loader_test = DataLoader(
                dset_test,
                batch_size=1,#This is irrelative to the args batch size parameter
                shuffle =False,
                num_workers=1)



        #Defining the model 
        model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
        output_feat=args.output_size,seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()
        model.load_state_dict(torch.load(model_path))


        ade_ =999999
        fde_ =999999
        print("Testing ....")
        ad,fd,pred_c, gt_c, topk_ade, topk_fde= test()
        print("ADE:",ad," FDE:",fd, " Pred:", pred_c, " GT:", gt_c, " Top3 ADE:", topk_ade, " Top3 FDE:", topk_fde)




    print("*"*50)

    # print("Avg ADE:",sum(ade_ls)/5)
    # print("Avg FDE:",sum(fde_ls)/5)
