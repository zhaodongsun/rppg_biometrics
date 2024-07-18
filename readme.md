# Biometric Authentication based on Enhanced Remote Photoplethysmography Signal Morphology

Welcome to use the code from our paper "Biometric Authentication based on Enhanced Remote Photoplethysmography Signal Morphology". **The training weights, data, and results can be downloaded [here](https://1drv.ms/u/s!AtCpzthip8c9_X4BfGlm84YGt6g5?e=yZVaTn). Please unzip `rppg_biometrics_artifacts.zip` to the main folder.**

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

Weights and irrelevant power ratios (IPR) are saved during training at `./results` (The training records are already in the folder.). The best weight is chosen at the lowest IPR by `./notebooks/sacred_train.ipynb`. The chosen weight are stored at `./rppg_model_pretrained_weights.pt`, which will be used in the 2nd training stage.

### The 2nd training stage: rPPG-cPPG Hybrid Training

if `./data_example/h5_obf` contains the complete OBF data and `./data_example/external_cppg.h5` exists, one can run `python joint_rppg_cppg_hybrid_training.py` to start the 2nd training stage. More details can be found in the comments of the `.py` file. Weights are saved during training. `joint_inference.py` is used to get the ID prediction results and the rPPG signals at different epochs as shown below.

```
for epoch in {0..29}
do
    python joint_inference.py with train_exp_num=1 e=$(awk "BEGIN {print(${epoch}*100)}") -i $(awk "BEGIN {print(${epoch}*100)}")
done
```

The training records after the 2nd training stage are in `./joint_results/default/1`. One can run `python eval_eer_auc.py` to get the EER and AUC metrics. One can run `python eval_morph.py` to get the Pearson correlation results for the morphology evaluation. More details can be found in the comments of the `.py` files.

## Citation
```
@article{sun2024biometrics,
  title={Biometric Authentication Based on Enhanced Remote Photoplethysmography Signal Morphology},
  author={Sun, Zhaodong and Li, Xiaobai and Komulainen, Jukka and Zhao, Guoying},
  booktitle={International Joint Conference on Biometrics (IJCB)},
  year={2024},
}
```
