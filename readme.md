# Privacy-preserving Face Biometrics based on Enhanced Remote Photoplethysmography Signal Morphology

Welcome to use the code from our paper.

## Requirement

Please see `requirement.txt` for the required python environment.

## Dataset

The data structures are as follows.

```
data_example
├── h5_obf # this folder contain data examples from OBF dataset
    ├── 001_xx_1.h5 # 001 means the subject ID number. 1 means the first session (pre-exercise)
        ├── imgs # shape (T, 6, 6, 3), the downsampled videos frames,permutation will be performaned during loading data. T is the number of frames, 6 is the spatial dimension, 3 is the RGB channels.
        ├── bvp # shape (T,), the ground truth cPPG signal. NOT USED DURING TRAINING. only used for morphology evaluation.
    ├── 001_xx_2.h5 # 001 means the subject ID number. 2 means the second session (post-exercise)
    .
    ├── 100_xx_1.h5
    ├── 100_xx_2.h5

├── external_cppg.h5 # A combination of cPPG biometrics datasets including Biosec2, BIDMC, and PRRB for rPPG-cPPG hybrid training. 
```




## Training

### The 1st training stage: rPPG Unsupervised Pre-training

if `./data_example/h5_obf` contains the complete OBF data, one can run `python rppg_model_pretraining.py` to start the 1st training stage. More details can be found in the comments of the `.py` file.

Weights and irrelevant power ratios (IPR) are saved during training. The best weight is chosen at the lowest IPR. The chosen weight should be stored at `./rppg_model_pretrained_weights.pt`, which will be used in the 2nd training stage.

### The 2nd training stage: rPPG-cPPG Hybrid Training

if `./data_example/h5_obf` contains the complete OBF data and `./data_example/external_cppg.h5` exists, one can run `python joint_rppg_cppg_hybrid_training.py` to start the 2nd training stage. More details can be found in the comments of the `.py` file. Weights are saved during training. `joint_inference.py` is used to get the ID prediction results and the rPPG signals.

## Results

The results after the 2nd training stage are in `./joint_results/default/1/2700`, which is chosen from the best epoch in the validation set. The folder contains `.npz` files which contain rPPG periodic segments, cPPG periodic segments, and classification outputs. 

One can run `python eval_eer_auc.py` to replicate the EER and AUC results. One can run `python eval_morph.py` to replicate the Pearson correlation results for the morphology evaluation. More details can be found in the comments of the `.py` files.

## Citation
```
@article{sun2024biometrics,
  title={Biometric Authentication Based on Enhanced Remote Photoplethysmography Signal Morphology},
  author={Sun, Zhaodong and Li, Xiaobai and Komulainen, Jukka and Zhao, Guoying},
  booktitle={International Joint Conference on Biometrics (IJCB)},
  year={2024},
}
```
