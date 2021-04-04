import json
from TypeNet import Resnest50
import os
import torch
from Dataset import get_dataloader
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

files = ['train_1', 'train', 'test']
IMG_MEAN = [0.49051854764297603, 0.41864813159033654, 0.3691471953573637]
IMG_STD = [0.2694104005082561, 0.24670860357502575, 0.29475671020139316]


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


if __name__ == '__main__':
    DEVICE = torch.device("cuda:3")
    model = Resnest50(15)
    model.load_state_dict(
        torch.load('checkpoints/'
                   'Resnest50-epoch_50-bs_64-lr_1e4-wd_1e3-momentum_0.9-df_15/model_epoch_50.ckpt',
                   map_location=DEVICE)['state_dict']
    )
    model.to(DEVICE)
    model.eval()
    train_loader, val_loader, test_loader = get_dataloader(None)
    loaders = [train_loader, val_loader, test_loader]
    path = '../Features/'
    with open(path + 'BodyFeature_Guo.json', 'r') as f:
        bf = json.load(f)

    Features = {}
    for loader, file in zip(loaders, files):
        cnt = 0
        jsonPath = path + 'FeaturesInGuo_{}.json'.format(file)
        pred = []
        targ = []
        if file == 'test':
            Features = {}
        for (datas, img_paths, img_names), (sexs, targets) in loader:
            for data, img_path, img_name, sex, target in zip(datas, img_paths, img_names, sexs, targets):
                cnt += 1
                values = {}

                # df
                data, target = torch.unsqueeze(data.to(DEVICE), 0), target.to(DEVICE)
                conv_out = LayerActivations(model.fc[0], None)
                out = model(data)
                pred.append(out.item())
                targ.append(target.item())
                conv_out.remove()
                xs = torch.squeeze(conv_out.features.cpu().detach()).numpy()
                values['deep features'] = xs.tolist()

                # sf
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

                # bf
                values['WSR'] = bf[img_name]['WSR']
                values['WTR'] = bf[img_name]['WTR']
                values['WHpR'] = bf[img_name]['WHpR']
                values['WHdR'] = bf[img_name]['WHdR']
                values['HpHdR'] = bf[img_name]['HpHdR']
                values['Area'] = bf[img_name]['Area']
                values['H2W'] = bf[img_name]['H2W']
                values['BMI'] = bf[img_name]['BMI']
                values['Sex'] = bf[img_name]['Sex']

                Features[img_name] = values

        if file == 'train_1':
            continue
        json_str = json.dumps(Features)
        with open(jsonPath, 'w') as json_file:
            json_file.write(json_str)

        MAE = mean_absolute_error(targ, pred)
        print('Correction: ', 'Resnest', file, ' MAE: ', MAE, 'image_nums: ', cnt)

    with open('../Features/FeaturesInGuo_train.json', 'r') as f:
        feas = json.load(f)
