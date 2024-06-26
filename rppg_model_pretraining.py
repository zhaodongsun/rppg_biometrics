import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import h5py
import torch
from rppg_model import rppg_model
from rppg_model_loss import ContrastLoss
from IrrelevantPowerRatio import IrrelevantPowerRatio

from utils_data import *
from utils_sig import *
from torch import optim
from torch.utils.data import DataLoader
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('model_train', save_git_info=False)


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

@ex.config
def my_config():
    # hyperparameters

    total_epoch = 30 # total number of epoch
    lr = 1e-5 # learning rate
    in_ch = 3 #number of input video channels, in_ch=3 for RGB videos

    fs = 60 # video frame rate
    T = fs * 10 # input video length, default 10s.

    # hyperparams for rPPG spatiotemporal sampling
    delta_t = int(T/2) # time length of each rPPG sample
    K = 4  # the number of rPPG samples per row of an rPPG ST map
    
    train_exp_name = 'default'
    result_dir = './results/%s'%(train_exp_name) # store checkpoints and training recording
    os.makedirs(result_dir, exist_ok=True)
    ex.observers.append(FileStorageObserver(result_dir))

@ex.automain
def my_main(_run, total_epoch, T, lr, result_dir, fs, delta_t, K, in_ch):

    exp_dir = result_dir + '/%d'%(int(_run._id)) # store experiment recording to the path

    # training list.
    train_list = glob.glob('./data_example/h5_obf/*1.h5') # train on pre-exercise videos. During loading data, the first 60% (3min out of 5min) length of each video is used for training.
    np.save(exp_dir+'/train_list.npy', train_list)

    # define the dataloader
    dataset = H5Dataset(train_list, T)
    dataloader = DataLoader(dataset, batch_size=2, # two videos for contrastive learning
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    # define the model and loss
    model = rppg_model(fs).to(device).train()
    loss_func = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250)

    # define irrelevant power ratio
    IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    # define the optimizer
    opt = optim.AdamW(model.parameters(), lr=lr)

    for e in range(total_epoch):
        for it in range(np.round(180/(T/fs)).astype('int')): # 180 means the video length of each video is 180s (3min).
            for imgs in dataloader: # dataloader randomly samples a video clip with length T
                imgs = imgs.to(device)
                
                # model forward propagation
                model_output, rppg = model(imgs) # model_output is the rPPG ST map

                # define the loss functions
                loss, p_loss, n_loss = loss_func(model_output)

                # optimize
                opt.zero_grad()
                loss.backward()
                opt.step()

                # evaluate irrelevant power ratio during training
                ipr = torch.mean(IPR(rppg.clone().detach()))

                # save loss values and IPR
                ex.log_scalar("loss", loss.item())
                ex.log_scalar("p_loss", p_loss.item())
                ex.log_scalar("n_loss", n_loss.item())
                ex.log_scalar("ipr", ipr.item())

        # save model checkpoints
        torch.save(model.state_dict(), exp_dir+'/epoch%d.pt'%e)