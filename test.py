"""
Testing code for the paper:
Autonomous Quality and Hallucination Assessment for Virtual Tissue Staining and Digital Pathology
Luzhe Huang, Yuzhu Li et al.
All rights reserved. Do not distribute without permission.
"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import pandas as pd
import pickle


# set path
save_model_path = "./ckpts/demo_model_T=5"
save_res_path = './predictions/demo_model_T=5'
N_REP = 10  # number of repetitions/ensembles
VOTERS = [0, 1, 2, 3, 4]  # indices of the optimal combination of ensembles
model_ckpts = ['best']

# SET TEST PATH
neg_ckpts = [1098]  # index of negative models, labeled as 1s
pos_ckpts = [54]  # index of positive models, labeled as 0s
valid_path = "./demo_data/test_vs"


# training parameters
k = 2           # number of target category
res_size = 712

# Select which frame to begin & end in cycles
begin_frame, end_frame, skip_frame = 0, 5, 1

# data loading parameters
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# SET DATALOADER PARAMETERS
params = {'batch_size': 4, 'shuffle': True, 'num_workers': 2, 'pin_memory': True, 'drop_last': False} if use_cuda else {}

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()


if VOTERS is None:
    VOTERS = list(range(0, N_REP))
for i in VOTERS:
    save_path = save_res_path+'_%d'%(i+1)
    model_path = save_model_path+'_%d'%(i+1)
    for mckpt in model_ckpts:
        res_path = os.path.join(save_path, str(mckpt))
        os.makedirs(res_path, exist_ok=True)

        for ckpt in neg_ckpts:
            valid_set = CCUQDataset(valid_path+'_%d'%ckpt, None, selected_frames, ch_per_img=3)

            # reset data loader
            all_data_loader = data.DataLoader(valid_set, **params)

            # reload CRNN model
            # create model
            cnn3d = ResNet_rep(in_channels=3, in_imgs=2, num_rep=len(selected_frames), num_classes=k).to(device)

            # Parallelize model to multiple GPUs
            if torch.cuda.device_count() > 1:
                print("Using", torch.cuda.device_count(), "GPUs!")
                cnn3d = nn.DataParallel(cnn3d)

            if mckpt == 'best':
                cnn3d.load_state_dict(torch.load(os.path.join(model_path, '3dcnn_best.pth')))
            else:
                cnn3d.load_state_dict(torch.load(os.path.join(model_path, '3dcnn_epoch%d.pth'%mckpt)))
            print('CNN3D model reloaded!')

            total_param = 0
            for p in cnn3d.parameters():
                if p.requires_grad:
                    total_param += p.numel()
            print("Total trainable parameters %d" % total_param)

            # make all video predictions by reloaded model
            print('Predicting all {} data:'.format(len(all_data_loader.dataset)))

            all_y_prob, all_y_list = Conv3d_final_prediction(cnn3d, device, all_data_loader, prob=True)
            all_y_pred = np.argmax(all_y_prob, axis=1)
            acc = accuracy_score(all_y_list, all_y_pred)

            print('Overall accuracy of Conv3D model at checkpoint %d: %f' % (ckpt, acc))

            # write in pandas dataframe
            df = pd.DataFrame(data={'filename': valid_set.flist, 'y': all_y_list, 'y_pred': all_y_pred, 'y_prob': all_y_prob[:, 1]})
            df.to_csv(os.path.join(res_path, 'AQUA_prediction_%d.csv'%ckpt))  # save pandas dataframe

        for ckpt in pos_ckpts:
            valid_set = CCUQDataset(None, valid_path+'_%d'%ckpt, selected_frames, ch_per_img=3)

            # reset data loader
            all_data_loader = data.DataLoader(valid_set, **params)

            # reload CRNN model
            # create model
            cnn3d = ResNet_rep(in_channels=3, in_imgs=2, num_rep=len(selected_frames), num_classes=k).to(device)

            # Parallelize model to multiple GPUs
            if torch.cuda.device_count() > 1:
                print("Using", torch.cuda.device_count(), "GPUs!")
                cnn3d = nn.DataParallel(cnn3d)

            if mckpt == 'best':
                cnn3d.load_state_dict(torch.load(os.path.join(model_path, '3dcnn_best.pth')))
            else:
                cnn3d.load_state_dict(torch.load(os.path.join(model_path, '3dcnn_epoch%d.pth'%mckpt)))
            print('CNN3D model reloaded!')

            total_param = 0
            for p in cnn3d.parameters():
                if p.requires_grad:
                    total_param += p.numel()
            print("Total trainable parameters %d" % total_param)

            # make all video predictions by reloaded model
            print('Predicting all {} data:'.format(len(all_data_loader.dataset)))

            all_y_prob, all_y_list = Conv3d_final_prediction(cnn3d, device, all_data_loader, prob=True)
            all_y_pred = np.argmax(all_y_prob, axis=1)
            acc = accuracy_score(all_y_list, all_y_pred)

            print('Overall accuracy of Conv3D model at checkpoint %d: %f' % (ckpt, acc))

            # write in pandas dataframe
            df = pd.DataFrame(data={'filename': valid_set.flist, 'y': all_y_list, 'y_pred':  all_y_pred, 'y_prob': all_y_prob[:, 1]})
            df.to_csv(os.path.join(res_path, 'AQUA_prediction_%d.csv'%ckpt))  # save pandas dataframe

