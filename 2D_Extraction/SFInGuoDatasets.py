import sys
import os
import numpy as np
import torch
from datasets import get_dataloader
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA


def Entropy(img):
    value, counts = np.unique(img, return_counts=True)
    freq_counts = counts / counts.sum()
    return -((freq_counts * np.log(freq_counts)).sum())


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloader(None)
    StatisticsFeature = {}
    cnt = 0
    loaders = [train_loader, val_loader, test_loader]
    pca_data = []
    for loader in loaders:
        for (data, name_path, name), (sex, bmi) in loader:
            values = {}
            x_data = torch.squeeze(data).reshape(3, -1).numpy()*255
            x_data = x_data[x_data != 0]
            values['Max'] = int(np.max(x_data, axis=0))
            values['Mode'] = int(stats.mode(x_data)[0][0])
            values['Range'] = int(np.max(x_data)-np.min(x_data))
            values['Entropy'] = Entropy(x_data)
            values['Mean'] = np.mean(x_data)
            values['Variance'] = np.var(x_data)
            values['Skewness'] = stats.skew(x_data)
            values['Kurtosis'] = stats.kurtosis(x_data)
            pca_data.append(torch.squeeze(data).reshape(-1).numpy()*255)
            print(1)
            break
        break
    # print(x_data.shape)
    pca = PCA(n_components=2)
    component = pca.fit_transform(pca_data)
    print(component)
    print(pca.explained_variance_ratio_)

    