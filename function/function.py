import torch
import torch.nn as nn
import random
import numpy as np
import cv2
import librosa
from torch.optim import lr_scheduler
from random import randint
import time
from tqdm import tqdm
import os
import torch.optim as optim

#-------------SEED--------------------------------------#
def seed_func():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#-------------Spectrogram--------------------------------#

def to_spectrum(data):
    spectrograms = []

    for i in range(data.shape[0]):
        signal = data[i, :]
        signal = np.array(signal)
        spectrogram = librosa.stft(signal, n_fft=512, hop_length=512)
        spectrogram = np.abs(spectrogram)**2
        log_spectrogram = librosa.power_to_db(spectrogram)
        log_spectrogram = cv2.resize(log_spectrogram, (64, 64))
        spectrograms.append(log_spectrogram)

    data = np.stack(spectrograms).astype(np.float32)
    data = torch.from_numpy(data).unsqueeze(dim = 1)

    return data

#-------------For Fewshot Learning-----------------------#

def convert_for_5shots(support_images, support_targets, device):

    support_targets = support_targets.cpu()
    labels = torch.unique(support_targets)
    new_support_images = []

    for label in labels:
        label_images = support_images[:, support_targets[0] == label]
        padded_label_images = torch.zeros((10, 1, 64, 64), dtype=label_images.dtype)
        padded_label_images[:label_images.shape[1]] = label_images.squeeze(0)
        new_support_images.append(padded_label_images.to(device))

    return new_support_images

#---------------------------------------Calculate accuracy-----------------------
def cal_accuracy_fewshot(loader, net, device):
    true_label = 0
    num_batches = 0

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = support_images.permute(1, 0, 2, 3, 4).to(device)
        targets = query_targets.to(device)
        targets = targets.permute(1,0)

        for i in range(len(q)):
            scores = net(q[i], s).float()
            target = targets[i].long()
            true_label += 1 if torch.argmax(scores) == target else 0
            num_batches += 1

    return true_label/num_batches

#-------------------------Calculate accuracy Ensemble share weight---------------------#
def cal_accuracy_fewshot_ensemble_1shot(loader, net, device):
    true_label = 0
    num_batches = 0

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = support_images.permute(1, 0, 2, 3, 4).to(device)
        targets = query_targets.to(device)
        targets = targets.permute(1,0)

        for i in range(len(q)):
            _,_,scores = net(q[i], s)
            target = targets[i].long()
            true_label += 1 if torch.argmax(scores) == target else 0
            num_batches += 1

    return true_label/num_batches

def cal_accuracy_fewshot_ensemble_5shot(loader, net, device):
    true_label = 0
    num_batches = 0

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4).to(device)
        s = convert_for_5shots(support_images, support_targets, device)
        targets = query_targets.to(device)
        targets = targets.permute(1,0)

        for i in range(len(q)):
            _,_,scores = net(q[i], s)
            target = targets[i].long()
            true_label += 1 if torch.argmax(scores) == target else 0
            num_batches += 1

    return true_label/num_batches

#------------------Predict fewshot-----------------------------------------------#
def predicted_fewshot(loader, net, device):
    predicted = []
    true_labels = []

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4)
        s = support_images.permute(1, 0, 2, 3, 4)
        targets = query_targets.to(device)
        targets = targets.permute(1,0)

        for i in range(len(q)):
            scores = net(q[i], s).float()
            target = targets[i].long()
            predicted.append(scores.cpu().detach().numpy())
            true_labels.append(target.cpu().detach().numpy())

    return np.array(true_labels), np.array(predicted)

def predicted_fewshot_ensemble(loader, net, device):
    predicted = []
    true_labels = []

    for query_images, query_targets, support_images, support_targets in loader:
        q = query_images.permute(1, 0, 2, 3, 4)
        s = support_images.permute(1, 0, 2, 3, 4)
        targets = query_targets.to(device)
        targets = targets.permute(1,0)

        for i in range(len(q)):
            _,_,scores = net(q[i], s)
            target = targets[i].long()
            predicted.append(scores.cpu().detach().numpy())
            true_labels.append(target.cpu().detach().numpy())

    return np.array(true_labels), np.array(predicted)

#-------------------Contrastive Loss------------------------#
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    def forward(self, ouput, target):
        upper = torch.exp(ouput[:,target])
        lower = torch.exp(ouput).sum(1)
        loss = -torch.log(upper / lower)

        return loss

#--------------------print model-------------------------------------
def print_model_layers(model):
    print("Model Layers:")
    print("=" * 70)
    for name, module in model.named_modules():
        if isinstance(module, nn.Module):
            print(f"{name}:")
            print(module)
            print("-" * 70)
    print("=" * 70)