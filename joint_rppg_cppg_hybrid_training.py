import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import h5py
import torch
import torch.nn as nn
from rppg_model import rppg_model
from biometric_models import *
from cycle_cut import cycle_cut
import matplotlib.pyplot as plt

from utils_data import *
from utils_sig import *
from torch import optim
from torch.utils.data import DataLoader
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('model_train', save_git_info=False)


if torch.cuda.is_available():
    device = torch.device('cuda')
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

@ex.config
def my_config():
    # hyperparameters

    # hyperparams for model training
    total_epoch = 3000 # total number of epochs for training the model
    batchsize = 16
    lr = 1e-3 # learning rate

    fs = 60 # video frame rate
    T = fs * 60 #  input length: 60 seconds.

    pretrain_rppg = './rppg_model_pretrained_weights.pt' # pretrained rPPG model weights from the 1st training stage

    num_classes_old = 195 # the number of subjects in external cPPG datasets
    num_classes = 100 # the number of subjects in OBF dataset (rPPG dataset)
    reverse = -1 
    # Since the rPPG model is trained unsupervised in the 1st training stage, rPPG signal could be reversed or not.
    # Green signals in facial videos are negatively correlated with rPPG signals.
    # We check the correlations between green signals and rPPG signals.
    # If rPPG signals positively correlate with green signals, rPPG signals should be reversed (reverse = -1).
    # If rPPG signals negatively correlate with green signals, rPPG signals should not be reversed (reverse = 1).

    train_exp_name = 'default'
    result_dir = './joint_results/%s'%(train_exp_name) # store checkpoints and training recording
    os.makedirs(result_dir, exist_ok=True)
    ex.observers.append(FileStorageObserver(result_dir))

@ex.automain
def my_main(_run, total_epoch, batchsize, T, lr, result_dir, fs, pretrain_rppg, num_classes_old, num_classes, reverse):

    exp_dir = result_dir + '/%d'%(int(_run._id)) # store experiment recording to the path

    # training list and inference list.
    train_list = glob.glob('./data_example/h5_obf/*1.h5') # train list on pre-exercise videos. # train on pre-exercise videos. During loading data, the first 60% (3min out of 5min) length of each video is used for training.
    test_list  = glob.glob('./data_example/h5_obf/*.h5') # inference list on pre-exercise and post-exercises videos, the following 20% length and the last 20% length of each video are used for validation and testing.

    
    np.save(exp_dir+'/train_list.npy', train_list)
    np.save(exp_dir+'/test_list.npy', test_list)

    # define the dataloader
    dataset = H5Dataset_id(train_list, T)
    dataloader = DataLoader(dataset, batch_size=batchsize,
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    # define the rPPG model
    model = rppg_model(fs).to(device).train()
    model.load_state_dict(torch.load(pretrain_rppg, map_location=device)) # load the rPPG model's weights from the 1st training stage

    # define the PPG-Morph model, the cPPG classification head is inside the ppg_model
    ppg_model = ppg_transformer(num_classes_old).to(device).train()
    # define the rPPG classification head
    cls_head = nn.Linear(64, num_classes).to(device).train()

    loss_func = nn.CrossEntropyLoss()

    # define the optimizer for the rPPG branch
    opt = optim.Adam(list(model.parameters())+list(cls_head.parameters())+list(ppg_model.parameters()), lr=lr) # TODO
    # define the optimizer for cPPG branch
    opt_ppg = optim.Adam(list(ppg_model.parameters()), lr=lr) # TODO

    # load the external cPPG biometric data
    # Note that './data_example/external_cppg.h5' is not uploaded due to the size limit of supplementary materials.
    with h5py.File('./data_example/external_cppg.h5', 'r') as f:
        ppg_clip = f['train']['ppg_clip'][:].astype('float32')
        ppg_clip = ppg_clip[:, np.newaxis]
        ppg_label = f['train']['label'][:]

    for e in range(total_epoch):
        for it in range(np.round(180/(T/fs)).astype('int')): ## 180 means the video length of each video is 180s (3min).
            for imgs, label in dataloader: # dataloader randomly samples a video clip with length T
                imgs = imgs.to(device) # ST map
                label = label.cpu().numpy() #id label

                ############### rPPG branch training ###############
                _, rppg = model(imgs)
                rppg = reverse * rppg # reverse rPPG is necessary, see more details from comments about reverse in the hyperparam part above.#####
                cycle_list = cycle_cut(rppg, fs, length=90) # cur the rPPG signal into periodic segments with length 90.
                if len(cycle_list)!=rppg.shape[0]:
                    ex.log_scalar("loss", 100)
                    ex.log_scalar("acc", 0)
                    ex.log_scalar("no_peak", 1)
                    continue
                cycles = torch.cat(cycle_list, 0)
                _, cycle_f = ppg_model(cycles) # get morph features from rPPG periodic segments
                pred_cls = cls_head(cycle_f) # ID prediction for rPPG periodic segments

                # prepare the GT ID labels for each segment
                cycle_label = []
                for c, l in zip(cycle_list, label):
                    cycle_label += [l]*c.shape[0]
                cycle_label = torch.tensor(cycle_label).to(device)
                assert cycle_label.shape[0]==cycles.shape[0]

                # cross-entropy for rPPG-ID loss
                loss = loss_func(pred_cls, cycle_label)
                
                # # optimize
                opt.zero_grad()
                loss.backward()
                opt.step()

                # # evaluate accuracy during training
                acc = np.mean(np.argmax(pred_cls.detach().cpu().numpy(), 1)==cycle_label.detach().cpu().numpy())
                # # save loss values
                ex.log_scalar("loss", loss.item())
                ex.log_scalar("acc", acc)
                ex.log_scalar("no_peak", 0)

                ############### cPPG branch training ###############
                ppg_choose = np.random.permutation(ppg_clip.shape[0])[:cycles.shape[0]]
                ppg_c = torch.tensor(ppg_clip[ppg_choose]).to(device) # choose cPPG periodic segments from external cPPG datasets
                ppg_l = torch.tensor(ppg_label[ppg_choose]).to(device) # ID labels for cPPG periodic segments

                ppg_pred_cls, _ = ppg_model(ppg_c) # ID prediction for cPPG periodic segments
                ppg_loss = loss_func(ppg_pred_cls, ppg_l) # cross-entropy for cPPG-ID loss

                opt_ppg.zero_grad()
                ppg_loss.backward()
                opt_ppg.step()
                
        # save model checkpoints
        if e%100==0:
            torch.save(model.state_dict(), exp_dir+'/epoch%d_model.pt'%e)
            torch.save(ppg_model.state_dict(), exp_dir+'/epoch%d_ppg_model.pt'%e)
            torch.save(cls_head.state_dict(), exp_dir+'/epoch%d_cls_head.pt'%e)