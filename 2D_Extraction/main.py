import torch.nn as nn
import sys
import torch.optim as optim
import torch
import numpy as np
import random
from torch.backends import cudnn
import argparse
from datasets import get_dataloader
from sklearn.metrics import mean_absolute_error
# from model import BFDFNet
from train import Trainer
from scipy import stats
import os
from ruamel import yaml
from TypeNet import *
import warnings
from DeepFeatureExtractor import DeepFeatureExtractor
from Regression import Regression
import json

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch Feature Comparison')
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def setup_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# hyper parameter
config_dict = {
    'dataset_root': '../Datasets/RGB/',
    'file_root': 'checkpoints/',
    'checkpoint': 'DF-40',
    'model_name': 'Resnest50',
    'workers': 4,
    'epochs': 50,
    'batch_size': 64,
    'lr': 1e-4,
    'momentum': 0.9,
    'weight_decay': 1e-3,
    'resume': '',
    'seed': 173,
    'gpu': '0',
    'df': 40,
}


def Entropy(img):
    value, counts = np.unique(img, return_counts=True)
    freq_counts = counts / counts.sum()
    return -((freq_counts * np.log(freq_counts)).sum())


def main():
    yamlpath = os.path.join('config',
                            config_dict['checkpoint'] + '.yaml')
    with open(yamlpath, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, Dumper=yaml.RoundTripDumper)

    with open(yamlpath, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    save_dir = config['file_root'] + config['checkpoint']
    resume = config['file_root'] + config['resume']
    setup_seed(config['seed'])  # 复现结果
    train_loader, val_loader, test_loader = get_dataloader(config)
    DEVICE = torch.device('cuda:' + config['gpu'])
    model = eval(config['model_name'])(df=config['df']).to(DEVICE)
    optimizer = optim.Adam\
        (filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'],
         weight_decay=config['weight_decay'])
    criterion = nn.MSELoss().to(DEVICE)
    Train = Trainer(model, DEVICE, optimizer, criterion, visual_path=config['checkpoint'], save_dir=save_dir,)
    Train.Loop(epochs=config["epochs"], trainloader=train_loader, testloader=val_loader)
    Train.test(test_loader, sex='diff')
    model.eval()


if __name__ == '__main__':
    main()
