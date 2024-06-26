import numpy as np
import h5py
import torch
from rppg_model import rppg_model
from biometric_models import *
from cycle_cut import cycle_cut
from utils_data import *
from utils_sig import *
from sacred import Experiment
from sacred.observers import FileStorageObserver
import json

ex = Experiment('model_pred', save_git_info=False)

@ex.config
def my_config():
    e = 2990 # the model checkpoint at epoch e
    train_exp_name = 'default'
    train_exp_num = 1 # the training experiment number
    train_exp_dir = './joint_results/%s/%d'%(train_exp_name, train_exp_num) # training experiment directory

    ex.observers.append(FileStorageObserver(train_exp_dir))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    else:
        device = torch.device('cpu')

@ex.automain
def my_main(_run, e, train_exp_dir, device):

    # load test file paths
    test_list = list(np.load(train_exp_dir + '/test_list.npy'))
    pred_exp_dir = train_exp_dir + '/%d'%(int(_run._id)) # prediction experiment directory

    with open(train_exp_dir+'/config.json') as f:
        config_train = json.load(f)

    model = rppg_model(config_train['fs']).to(device).eval()
    model.load_state_dict(torch.load(train_exp_dir+'/epoch%d_model.pt'%(e), map_location=device)) # load weights to the model
    ppg_model = ppg_transformer(config_train['num_classes_old']).to(device).train()
    ppg_model.load_state_dict(torch.load(train_exp_dir+'/epoch%d_ppg_model.pt'%(e), map_location=device))
    cls_head = nn.Linear(64, config_train['num_classes']).to(device).train()
    cls_head.load_state_dict(torch.load(train_exp_dir+'/epoch%d_cls_head.pt'%(e), map_location=device))

    @torch.no_grad()
    def dl_model(imgs_clip, fs):
        # model inference
        img_batch = imgs_clip
        img_batch = img_batch.transpose((3,0,1,2))

        # permutation
        T = img_batch.shape[1]
        hw = img_batch.shape[2]
        img_batch = img_batch.reshape(3, T, -1) 
        img_batch = img_batch[:,:,np.random.permutation(hw*hw)]
        img_batch = np.transpose(img_batch, (0,2,1)) # shape (3, N, T) 

        img_batch = img_batch[np.newaxis].astype('float32')
        img_batch = torch.tensor(img_batch).to(device)

        _, rppg = model(img_batch)
        rppg = config_train['reverse'] * rppg
        cycle_list = cycle_cut(rppg, fs, length=90) # cycle
        cycles = torch.cat(cycle_list, 0)
        _, cycle_f = ppg_model(cycles)
        pred_cls = cls_head(cycle_f)
        return rppg[0].detach().cpu().numpy(), cycles.detach().cpu().numpy(), pred_cls.detach().cpu().numpy()

    for h5_path in test_list:
        h5_path = str(h5_path)

        with h5py.File(h5_path, 'r') as f:
            imgs = f['imgs']
            bvp = f['bvp']
            fs = config_train['fs']

            img_length = np.min([imgs.shape[0], bvp.shape[0]])

            rppg_list = []
            cyc_list = []
            pred_list = []

            bvp_list = []
            bvp_cyc_list = []

            for b in range(1):
                rppg_sig, cyc, pred = dl_model(imgs[:img_length], fs)
                rppg_list.append(rppg_sig)
                cyc_list.append(cyc)
                pred_list.append(pred)

                bvp_clip = butter_bandpass(bvp[:img_length], lowcut=0.6, highcut=4, fs=fs).copy()
                bvp_list.append(bvp_clip)
                bvp_cyc_list.append(cycle_cut(torch.tensor(bvp_clip[np.newaxis, :]), fs, length=90)[0].cpu().numpy())

            rppg_list = np.array(rppg_list)
            cyc_list = np.array(cyc_list)
            pred_list = np.array(pred_list)

            bvp_list = np.array(bvp_list)
            bvp_cyc_list = np.array(bvp_cyc_list)

            results = {'rppg_list': rppg_list, 'bvp_list': bvp_list, 'bvp_cyc_list':bvp_cyc_list, 'cyc_list': cyc_list, 'pred_list': pred_list}
            np.save(pred_exp_dir+'/'+h5_path.split('/')[-1][:-3], results)