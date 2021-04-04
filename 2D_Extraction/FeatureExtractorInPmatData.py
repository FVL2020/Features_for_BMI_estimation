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
    'dataset_root': '/home/benkesheng/Feature_Comparison_171/Feature_Comparison/PmatData_Supine/',
    'file_root': '/home/benkesheng/Feature_Comparison_171/Feature_Comparison/DFExtraction/CNN/checkpoints/',
    'checkpoint': 'Pressure_Map',
    'model_name': 'Resnest50',
    'workers': 4,
    'epochs': 50,
    'batch_size': 64,
    'lr': 1e-4,
    'momentum': 0.9,
    'weight_decay': 1e-3,
    'resume': '',
    'seed': 173,
    'gpu': '3',
    'df': 10,
}
files = ['train_1', 'train', 'test']
class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()

def Entropy(img):
    value, counts = np.unique(img, return_counts=True)
    freq_counts = counts / counts.sum()
    return -((freq_counts * np.log(freq_counts)).sum())
def main():
    yamlpath = os.path.join('/home/benkesheng/Feature_Comparison_171/Feature_Comparison/DFExtraction/CNN/config',
                            config_dict['checkpoint'] + '.yaml')
    with open(yamlpath, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, Dumper=yaml.RoundTripDumper)

    with open(yamlpath, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    save_dir = config['file_root'] + config['checkpoint']
    resume = config['file_root'] + config['resume']
    setup_seed(config['seed'])  # 复现结果
    train_loader, val_loader, test_loader = get_dataloader(config)
    loaders = [train_loader, val_loader, test_loader]
    DEVICE = torch.device('cuda:'+config['gpu'])
    model = eval(config['model_name'])(df=config['df']).to(DEVICE)
    model.load_state_dict(
        torch.load('/home/benkesheng/Feature_Comparison_171/Feature_Comparison'
                   '/DFExtraction/CNN/checkpoints/Pressure_Map/model_epoch_50.ckpt',
                   )['state_dict']
    )
    model.eval()
    path = '/home/benkesheng/Feature_Comparison_171/Feature_Comparison/'
    with open(path + 'BodyFeature_PressureMap.json', 'r') as f:
        bf = json.load(f)

    Features = {}
    for loader, file in zip(loaders, files):
        cnt = 0
        jsonPath = path + 'FeaturesInPressureMap_{}.json'.format(file)
        pred = []
        targ = []
        if file == 'test':
            Features = {}
        for (datas,img_names), targets in loader:
            for data,img_name, target in zip(datas,img_names, targets):
                cnt += 1
                values = {}

                data, target = torch.unsqueeze(data.to(DEVICE), 0), target.to(DEVICE)
                conv_out = LayerActivations(model.fc[0], None)
                out = model(data)
                pred.append(out.item())
                targ.append(target.item())
                conv_out.remove()
                xs = torch.squeeze(conv_out.features.cpu().detach()).numpy()
                values['deep features'] = xs.tolist()

                data = torch.squeeze(data).permute(1, 2, 0).cpu().numpy()
                data = data * IMG_STD + IMG_MEAN
                x_data = data.reshape(3, -1) * 255
                x_data = x_data[x_data > 0.001]
                values['Max'] = int(np.max(x_data, axis=0))
                values['Mode'] = int(stats.mode(x_data)[0][0])
                values['Range'] = int(np.max(x_data) - np.min(x_data))
                values['Entropy'] = Entropy(x_data).item()
                values['Mean'] = np.mean(x_data).item()
                values['Variance'] = np.var(x_data).item()
                values['Skewness'] = stats.skew(x_data)
                values['Kurtosis'] = stats.kurtosis(x_data)

                values['WSR'] = bf[img_name]['WSR']
                values['WTR'] = bf[img_name]['WTR']
                values['WHpR'] = bf[img_name]['WHpR']
                values['WHdR'] = bf[img_name]['WHdR']
                values['HpHdR'] = bf[img_name]['HpHdR']
                values['Area'] = bf[img_name]['Area']
                values['H2W'] = bf[img_name]['H2W']
                values['BMI'] = bf[img_name]['BMI']

                Features[img_name] = values
        if file == 'train_1':
            continue
        json_str = json.dumps(Features)
        with open(jsonPath, 'w') as json_file:
            json_file.write(json_str)

        MAE = mean_absolute_error(targ, pred)
        print('Correction: ', 'Resnest', file, ' MAE: ', MAE, 'image_nums: ', cnt)



if __name__ == '__main__':
    with open('/home/benkesheng/Feature_Comparison_171/Feature_Comparison/FeaturesInPressureMap_train.json', 'r') as f:
        feas = json.load(f)
