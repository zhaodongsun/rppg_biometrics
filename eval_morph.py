import numpy as np
import glob
from scipy import stats as st
from scipy.stats import pearsonr

sess = 1
partition = 'test' # val or test. using val partition to find the best epoch and use test partition on the best epoch
epoch_list = [0] #list(range(0,3000,100))

ps_epoch = []
for e in epoch_list:
    ps_list = []

    for sub in range(100):
        # after joint training
        results = np.load(glob.glob('./joint_results/default/1/%d/%03d*%d.np*'%(e, sub+1, sess))[0], allow_pickle=True).item()
        cyc_list = results['cyc_list'][0,:,0]
        bvp_cyc_list = results['bvp_cyc_list'][0,:,0]

        idx6, idx8 = int(cyc_list.shape[0]*0.6), int(cyc_list.shape[0]*0.8)
        idx6_, idx8_ = int(bvp_cyc_list.shape[0]*0.6), int(bvp_cyc_list.shape[0]*0.8)
        if partition=='val':
            cyc_list = cyc_list[idx6:idx8]
            bvp_cyc_list = bvp_cyc_list[idx6_:idx8_]
        elif partition=='test':
            cyc_list = cyc_list[idx8:]
            bvp_cyc_list = bvp_cyc_list[idx8_:]
        else:
            raise('error')

        cyc_list_mean = np.mean(cyc_list, 0)
        bvp_cyc_list_mean = np.mean(bvp_cyc_list, 0)
        ps_list.append(pearsonr(bvp_cyc_list_mean, cyc_list_mean)[0])

    print(e, np.mean(ps_list))
    ps_epoch.append(np.mean(ps_list))

print('best:', epoch_list[np.argmax(ps_epoch)], ps_epoch[np.argmax(ps_epoch)])
