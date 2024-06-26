import neurokit2 as nk
import numpy as np
from torch.nn.functional import interpolate
import torch

def get_peak(sig, sig_fps):
    ppg_clean = nk.ppg_clean(sig.copy(), sampling_rate=sig_fps, method='elgendi')
    info = nk.ppg_findpeaks(ppg_clean,sampling_rate=sig_fps)
    peak_loc = info["PPG_Peaks"]
    return peak_loc


def cubic_resample(sig, length):
    sig = sig.unsqueeze(0).unsqueeze(0)
    sig_interp = interpolate(sig, size=length, mode='linear') # TODO
    return sig_interp

def get_cycle(sig, peak_loc, length):
    clip_list = []
    for k in np.arange(len(peak_loc)-1):
        sig_clip = sig[peak_loc[k]:peak_loc[k+1]]
        if sig_clip.std()==0:
            continue
        sig_clip = (sig_clip - sig_clip.mean()) / sig_clip.std()
        sig_clip = cubic_resample(sig_clip, length) # 90 cubic
        clip_list.append(sig_clip)
    return torch.cat(clip_list, 0)

def cycle_cut(sigs, fs, length=90):
    # sig shape: (N, T)
    cycle_list = []
    for s in sigs:
        peak_loc = get_peak(s.detach().cpu().numpy(), fs)
        try:
            ppg_clips = get_cycle(s, peak_loc, length)
        except:
            return []
        cycle_list.append(ppg_clips)
    return cycle_list

# def cycle_cut(sigs, bin_peak, fs, length=90):
#     # sig shape: (N, T)
#     cycle_list = []
#     for s, bp in zip(sigs, bin_peak):
#         peak_loc = np.where(bp==1)[0]
#         ppg_clips = get_cycle(s, peak_loc, length)
#         cycle_list.append(ppg_clips)
#     return cycle_list