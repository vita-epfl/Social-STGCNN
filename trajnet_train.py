

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

from utils import * 
from metrics import * 
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from model import *

## TrajNet++
import trajnetplusplustools
from data_load_utils import prepare_data
from trajnet_utils import TrajectoryDataset
from trajnet_loader import trajnet_loader
from helper_models import DummyGCN

parser = argparse.ArgumentParser()

# Trajnet loader
parser.add_argument("--fill_missing_obs", default=1, type=int)
parser.add_argument("--keep_single_ped_scenes", default=1, type=int)

#Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='eth_data',
                    help='eth,hotel,univ,zara1,zara2')    

#Training specifc parameters
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=250,
                    help='number of epochs')  
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')        
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=150,
                    help='number of steps to drop the lr')  
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag',
                    help='personal tag for the model ')
parser.add_argument('--sample', type=float, default=1.0,
                    help="Dataset ratio to sample.")             
args = parser.parse_args()



print('*'*30)
print("Training initiating....")
print(args)


def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)

#Data prep     
args.obs_len = args.obs_seq_len
args.pred_len = args.pred_seq_len
norm_lap_matr = True

# Trajnet train loader
train_loader, _, _ = prepare_data(
    'datasets/' + args.dataset, subset='/train/', sample=args.sample
    )

traj_train_loader = trajnet_loader(
    train_loader, 
    args,
    drop_distant_ped=False,
    test=False,
    keep_single_ped_scenes=args.keep_single_ped_scenes,
    fill_missing_obs=args.fill_missing_obs,
    norm_lap_matr=norm_lap_matr
    )
traj_train_loader = tqdm(traj_train_loader)
traj_train_loader = list(traj_train_loader)

# Trajnet val loader
val_loader, _, _ = prepare_data(
    'datasets/' + args.dataset, subset='/val/', sample=args.sample
    )

traj_val_loader = trajnet_loader(
    val_loader, 
    args,
    drop_distant_ped=False,
    test=True,
    keep_single_ped_scenes=args.keep_single_ped_scenes,
    fill_missing_obs=args.fill_missing_obs,
    norm_lap_matr=norm_lap_matr
    )
traj_val_loader = tqdm(traj_val_loader)
traj_val_loader = list(traj_val_loader)

#Defining the model 

model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
output_feat=args.output_size,seq_len=args.obs_seq_len,
kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()


#Training settings 

optimizer = optim.SGD(model.parameters(),lr=args.lr)

if args.use_lrschd:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
    


checkpoint_dir = './checkpoint/'+args.tag+'/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
    


print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)

#Training 
metrics = {'train_loss':[],  'val_loss':[]}
constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999}

def train(epoch):
    global args, metrics, traj_train_loader
    model.train()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(traj_train_loader)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1


    for cnt,batch in enumerate(traj_train_loader): 
        batch_count+=1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        # The rest of the code assumes batch size in the 0-th dimension
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

        optimizer.zero_grad()
        #Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        # If there's only one pedestrian, handle it differently
        if V_obs_tmp.shape[3] == 1:
            single_ped_model = DummyGCN(model, args)
            V_pred = single_ped_model(
                obs_traj[0, ...], obs_traj_rel[0, ...], norm_lap_matr
                )
        else: 
            V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.permute(0, 2, 3, 1)
        
        V_tr = V_tr[0, ...]
        A_tr = A_tr[0, ...]
        V_pred = V_pred[0, ...]

        if args.batch_size == 1 or (batch_count%args.batch_size !=0 and cnt != turn_point):
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            loss.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch/batch_count)
            
    metrics['train_loss'].append(loss_batch/batch_count)
    



def vald(epoch):
    global args, metrics, traj_val_loader, constant_metrics
    model.eval()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(traj_val_loader)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1
    
    for cnt,batch in enumerate(traj_val_loader): 
        batch_count+=1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        # The rest of the code assumes batch size in the 0-th dimension
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
                obs_traj[0, ...], obs_traj_rel[0, ...], norm_lap_matr
                )
        else: 
            V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr[0, ...]
        A_tr = A_tr[0, ...]
        V_pred = V_pred[0, ...]

        if args.batch_size == 1 or (batch_count%args.batch_size !=0 and cnt != turn_point):
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            #Metrics
            loss_batch += loss.item()
            print('VALD:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)

    metrics['val_loss'].append(loss_batch/batch_count)
    
    if  metrics['val_loss'][-1]< constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] =  metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(),checkpoint_dir+'val_best.pth')  # OK


print('Training started ...')
for epoch in range(args.num_epochs):
    train(epoch)
    vald(epoch)
    if args.use_lrschd:
        scheduler.step()


    print('*'*30)
    print('Epoch:',args.tag,":", epoch)
    for k,v in metrics.items():
        if len(v)>0:
            print(k,v[-1])


    print(constant_metrics)
    print('*'*30)
    
    with open(checkpoint_dir+'metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)
    
    with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)  




