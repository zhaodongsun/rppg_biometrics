import numpy as np
import glob
from scipy import stats as st
from sklearn.metrics import roc_curve
from sklearn import metrics

avg = 20 # ID prediction score averaged on consecutive periodic segments
sess = 1 # test on sess 1 (intra-sess testing) or sess 2 (cross-sess testing)
partition = 'test' # val or test. using val partition to find the best epoch and use test partition on the best epoch
epoch_list = [2700] # list(range(1000,3000,100))


def compute_eer(gt, pred_prob):
    # the function use binary GT labels and predicted probabilities to calculate eer and auc
    fpr, tpr, thres = roc_curve(gt, pred_prob)
    auc = metrics.auc(fpr, tpr)
    fnr = 1 - tpr

    eer_thres = thres[np.nanargmin(np.absolute((fnr - fpr)))]

    eer1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer1 + eer2) / 2
    return eer, eer_thres, auc

eer_epoch = []
auc_epoch = []
for e in epoch_list:
    # load data
    pred_sub_list = []
    label_list = [] 
    for sub in range(100): # get the ID prediction scores and GT labels.
        # results = np.load(glob.glob('./joint_results/default/1/%d/%03d*%d.npy'%(e, sub+1, sess))[0], allow_pickle=True).item()
        results = np.load(glob.glob('./joint_results/default/1/%d/%03d*%d.npz'%(e, sub+1, sess))[0])
        pred_sub = results['pred_list'][0]

        idx6, idx8 = int(pred_sub.shape[0]*0.6), int(pred_sub.shape[0]*0.8)
        if partition=='val': # 60%-80% part for validation
            pred_sub = pred_sub[idx6:idx8]
        elif partition=='test': # 80%-100% part for testing
            pred_sub = pred_sub[idx8:]
        else:
            raise('error')
        pred_sub_list.append(pred_sub)
        label_list += pred_sub.shape[0]*[sub]
    pred_sub_list = np.concatenate(pred_sub_list, 0)
    label_list = np.array(label_list)

    # compute metrics
    eer_list = []
    auc_list = []
    for s in range(100): # calculate eer for each subject
        pred_list_bin = []
        true_list_bin = []
        for sub in range(100): # convert to binary labels using one-vs-rest
            pred_sub = pred_sub_list[np.where(label_list==sub)]
            for i in range(0, pred_sub.shape[0], avg):
                # average consecutive periodic segments to get the classification probability
                pred_sub_ = pred_sub[i:i+avg]
                cls_prob = np.mean(pred_sub_, 0)

                # [0,1] prediction and binary labels
                pred_list_bin.append(cls_prob[s])
                true_list_bin.append(sub==s)

        eer, _, auc = compute_eer(true_list_bin, pred_list_bin)
        eer_list.append(eer)
        auc_list.append(auc)

    print(e, np.mean(eer_list), np.mean(auc_list))
    eer_epoch.append(np.mean(eer_list))
    auc_epoch.append(np.mean(auc_list))

print('best:', epoch_list[np.argmin(eer_epoch)], eer_epoch[np.argmin(eer_epoch)], auc_epoch[np.argmin(eer_epoch)])
