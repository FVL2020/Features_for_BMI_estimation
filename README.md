## SEEING HEALTH WITH EYES: FEATURE COMBINATION FOR IMAGE-BASED HUMAN BMI ESTIMATION
This repository is the official Pytorch implementation for this paper.
### Introduction
Body Mass Index (BMI) is an important measurement of human obesity and health, which can provide useful information for plenty of practical purposes, such as monitoring, re-identification, and health care. Recently, some data-driven advances have been proposed to estimate BMI by 2D or 3D features from face images, frontal-body images and RGB-D images. However, due to the privacy issue or limitations of 3D cameras, the required data is hard to be obtained. More importantly, each of the previous works has only studied for a single type of features, hence it is worth investigating whether combinations of different features are more effective. To address this issue, we analyze the correlation of various features extracted from 2D body images with the estimated BMI, and then propose an accurate BMI estimation method with the optimal feature combination. Extensive experiments demonstrate that the proposed method outperforms these image-based BMI estimation methods which only utilize a single type feature in most cases.
![framework](https://user-images.githubusercontent.com/63050198/113507783-1aec2180-957f-11eb-9368-0cf1017626ff.jpg)
### Requirements
- cuda & cudnn
- Python 3
- Pytorch 1.4
- torchvision 0.4.2
- opendr == 0.77
- scikit-image
- deepdish>=0.3
- opencv-python
- absl-py
- ipdb
- skikit-learn 0.22.1

### Tips
- The process of Anthropometric Features extraction is showed in other paper repository, We only have JSON files in our repository.
- You can download the 3D Features extracted model by 
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
```
