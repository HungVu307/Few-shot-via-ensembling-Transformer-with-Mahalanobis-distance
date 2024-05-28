import torch
import numpy as np
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import function.function as function
import time
from tqdm import tqdm
import os
from function.function import ContrastiveLoss, seed_func, cal_accuracy_fewshot_ensemble_1shot
from CWRU.CWRU_dataset import CWRU
from dataloader.dataloader import FewshotDataset
from torch.utils.data import DataLoader
from net.proposed_model import Ensemble_Net
from sklearn.metrics import confusion_matrix
import argparse
import torch.nn as nn
import numpy as np
import librosa
import cv2
import torch
from scipy.ndimage import gaussian_filter



parser = argparse.ArgumentParser(description='Bearing Faults Project Configuration')
parser.add_argument('--dataset', choices=['CWRU', 'PDB'], help='Dataset (Only CWRU or PDB)')
parser.add_argument('--training_samples_CWRU', type=int, default=30, help='Number of training samples for CWRU')
parser.add_argument('--training_samples_PDB', type=int, default=195, help='Number of training samples for PDB')
parser.add_argument('--model_name', type=str, help='Model name')
parser.add_argument('--episode_num_train', type=int, default=130, help='Number of training episodes')
parser.add_argument('--episode_num_test', type=int, default=150, help='Number of testing episodes')
parser.add_argument('--way_num_CWRU', type=int, default=10, help='Number of classes for CWRU')
parser.add_argument('--noise_DB', type=str, default=None, help='Noise database')
parser.add_argument('--way_num_PDB', type=int, default=13, help='Number of classes for PDB')
parser.add_argument('--spectrum', action='store_true', help='Use spectrum')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--path_weights', type=str, default='checkpoints/', help='Path to weights')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--loss1', default=ContrastiveLoss())
parser.add_argument('--loss2', default=nn.CrossEntropyLoss())
args = parser.parse_args()

print(args)
#---------------------------------------------------Load dataset-----------------------------------------------------------------------------------------:
if args.dataset == 'CWRU':
    window_size = 2048
    split = args.training_samples_CWRU//30
    data = CWRU(split, ['12DriveEndFault'], ['1772', '1750', '1730'], window_size)
    data.nclasses,data.classes,len(data.X_train),len(data.X_test)
    data.X_train = data.X_train.astype(np.float32)
    data.X_test = data.X_test.astype(np.float32)
    train_data_CWRU = torch.from_numpy(data.X_train)
    train_label_CWRU = torch.from_numpy(data.y_train)
    test_data_CWRU = torch.from_numpy(data.X_test)
    test_label_CWRU = torch.from_numpy(data.y_test)
    train_data_CWRU = train_data_CWRU.reshape([args.training_samples_CWRU,4096])
    test_data_CWRU = test_data_CWRU.reshape([750,4096])

    if args.noise_DB != None:
        snr_dB = args.noise_DB
        data.add_noise_to_test_data(snr_dB, 0.001)
        noisy_test_data = data.X_test_noisy.reshape([750,4096])

        if args.spectrum != None:
            train_data_CWRU = function.to_spectrum(train_data_CWRU)
            test_data_CWRU = function.to_spectrum(noisy_test_data)
        else:
            train_data_CWRU = train_data_CWRU.reshape(train_data_CWRU.shape[0], 1, 64, 64)
            test_data_CWRU = train_data_CWRU.reshape(test_data_CWRU.shape[0], 1, 64, 64)

    else:
        if args.spectrum != None:
            train_data_CWRU = function.to_spectrum(train_data_CWRU)
            test_data_CWRU = function.to_spectrum(test_data_CWRU)
        else:
            train_data_CWRU = train_data_CWRU.reshape(train_data_CWRU.shape[0], 1, 64, 64)
            test_data_CWRU = test_data_CWRU.reshape(test_data_CWRU.shape[0], 1, 64, 64)

    print('Shape of CWRU train data:',train_data_CWRU.shape)
    print('Shape of CWRU test data:',test_data_CWRU.shape)
    print('End Loading CWRU')
    
    train_dataset_CWRU = FewshotDataset(train_data_CWRU, train_label_CWRU, episode_num=args.episode_num_train, way_num=args.way_num_CWRU, shot_num=1, query_num=1)
    train_dataloader_CWRU = DataLoader(train_dataset_CWRU, batch_size=args.batch_size, shuffle=True)
    test_dataset_CWRU = FewshotDataset(test_data_CWRU, test_label_CWRU, episode_num=args.episode_num_test, way_num=args.way_num_CWRU, shot_num=1, query_num=1)
    test_dataloader_CWRU = DataLoader(test_dataset_CWRU, batch_size=args.batch_size, shuffle=False)


if args.dataset == 'PDB':
    def to_spectrum(data, h=64, w=64, sigma=0.6):
        spectrograms = []

        for i in range(data.shape[0]):
            signal = data[i, :]
            signal = np.array(signal)
            spectrogram = librosa.stft(signal, n_fft=512, hop_length=512)
            spectrogram = np.abs(spectrogram) ** 2
            log_spectrogram = librosa.power_to_db(spectrogram)
            log_spectrogram = cv2.resize(log_spectrogram, (h, w))
            smoothed_spectrogram = gaussian_filter(log_spectrogram, sigma=sigma)
            spectrograms.append(smoothed_spectrogram)

        data = np.stack(spectrograms).astype(np.float32)
        data = torch.from_numpy(data).unsqueeze(dim=1)

        return data
    print('Loading data................!!')
    data = torch.load('Paderborn/data_segment.pt')
    label = torch.load('Paderborn/label.pt')
    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    data = to_spectrum(data)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42, stratify=label)
    test_size = args.training_samples_PDB/x_train.shape[0]
    _, x_train_limited,_, y_train_limited = train_test_split(x_train, y_train, test_size=test_size, random_state=42, stratify=y_train)
    print('Data train for limited case shape:', x_train_limited.shape)
    print('Label train for limited case shape', y_train_limited.shape)
    train_dataset_PDB = FewshotDataset(x_train_limited, y_train_limited, episode_num=args.episode_num_train, way_num=args.way_num_PDB, shot_num=1, query_num=1)
    train_dataloader_PDB = DataLoader(train_dataset_PDB, batch_size=args.batch_size, shuffle=True)
    test_dataset_PDB = FewshotDataset(x_test, y_test, episode_num=args.episode_num_test, way_num=args.way_num_PDB, shot_num=1, query_num=1)
    test_dataloader_PDB = DataLoader(test_dataset_PDB, batch_size=args.batch_size, shuffle=False)


def train_and_test_model_ensemble(net,
                         train_dataloader,
                         test_loader,
                         training_samples,
                         num_epochs = args.num_epochs,
                         lr = args.lr,
                         loss1 = args.loss1,
                         loss2 = args.loss2,
                         path_weight = args.path_weights,
                         num_samples = args.training_samples_CWRU):
    device = args.device
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss1.to(device)
    loss2.to(device)
    full_loss = []
    full_acc = []
    pred_acc = 0

    cumulative_time = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        running_loss = 0
        num_batches = 0
        running_loss_1 = 0
        running_loss_2 = 0
        optimizer.zero_grad()
        print('='*50, 'Epoch:', epoch, '='*50)
        with tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs}', unit='batch') as t:
            for query_images, query_targets, support_images, support_targets in t:
                q = query_images.permute(1, 0, 2, 3, 4).to(device)
                s = support_images.permute(1, 0, 2, 3, 4).to(device)
                targets = query_targets.to(device)
                targets = targets.permute(1, 0)
                for i in range(len(q)):
                    m_l, m_u, _ = net(q[i], s)
                    target = targets[i].long()
                    loss = loss1(m_l, target) + loss2(m_u, target)
                    loss.backward()
                    running_loss += loss.detach().item()
                    running_loss_1 += loss1(m_l, target).detach().item()
                    running_loss_2 += loss2(m_u, target).detach().item()
                    num_batches += 1
                optimizer.step()
                optimizer.zero_grad()
                t.set_postfix(loss=running_loss / num_batches, loss_1 = running_loss_1 / num_batches, loss_2 = running_loss_2 / num_batches)

        elapsed_time = time.time() - start_time
        cumulative_time += elapsed_time
        cumulative_minutes = cumulative_time / 60
        print(f"Epoch {epoch}/{num_epochs} completed in {cumulative_minutes:.2f} minutes")

        scheduler.step()

        with torch.no_grad():
            total_loss = running_loss / num_batches
            full_loss.append(total_loss)
            print('------------Testing on the test set-------------')
            acc = cal_accuracy_fewshot_ensemble_1shot(test_loader, net, device)
            full_acc.append(acc)
            print(f'Accuracy on the test set: {acc:.4f}')
            if acc > pred_acc:
                if epoch >= 2:
                    os.remove(path_weight + model_name)
                pred_acc = acc
                model_name = f'{args.model_name}_1shot_{acc:.4f}_{training_samples}samples.pth'
                torch.save(net, path_weight + model_name)
                print(f'=> Save the best model with accuracy: {acc:.4f}')
        torch.cuda.empty_cache()

    return full_loss, full_acc


#----------------------------------------------------Training phase--------------------------------------------------#
seed_func()
net = Ensemble_Net()
net = net.to(args.device)
print('training.........................!!')
if args.dataset == 'CWRU':
    train_and_test_model_ensemble(net,
                        train_dataloader = train_dataloader_CWRU,
                        test_loader = test_dataloader_CWRU,
                        training_samples = args.training_samples_CWRU,
                        num_epochs = args.num_epochs,
                        lr = args.lr,
                        loss1 = args.loss1,
                        loss2 = args.loss2,
                        path_weight = args.path_weights,
                        num_samples = args.training_samples_CWRU)
elif args.dataset == 'PDB':
    train_and_test_model_ensemble(net,
                        train_dataloader = train_dataloader_PDB,
                        test_loader = test_dataloader_PDB,
                        training_samples = args.training_samples_PDB,
                        num_epochs = args.num_epochs,
                        lr = args.lr,
                        loss1 = args.loss1,
                        loss2 = args.loss2,
                        path_weight = args.path_weights,
                        num_samples = args.training_samples_PDB)
    print('end training...................!!')
