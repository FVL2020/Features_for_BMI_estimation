import torch.utils.data as data
import torch
from torchvision import transforms
import os
import re
import cv2
import warnings

warnings.filterwarnings("ignore")

# ImageNet
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# Author Dataset
# IMG_MEAN = [0.49051854764297603, 0.41864813159033654, 0.3691471953573637]
# IMG_STD = [0.2694104005082561, 0.24670860357502575, 0.29475671020139316]
IMG_SIZE = 224


def _get_image_size(img):
    if transforms.functional._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class Resize(transforms.Resize):
    def __call__(self, img):
        h, w = _get_image_size(img)
        scale = max(w, h) / float(self.size)
        new_w, new_h = int(w / scale), int(h / scale)
        return transforms.functional.resize(img, (new_w, new_h), self.interpolation)


def get_dataloader(args):
    if args['checkpoint'] == 'Pressure_Map':
        Dataset = pressure_map_datasets(args['dataset_root'])
        train_size = int(0.7 * len(Dataset))
        test_size = len(Dataset) - train_size
        val_size = int(train_size/7)
        train_size = train_size-val_size
        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
                Dataset,[train_size,test_size,val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                                   num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
        return train_loader, val_loader, test_loader

    if args is None:
        root = '../Datasets/RGB'
        train_dataset = datasets(root, 'Image_train')
        test_dataset = datasets(root, 'Image_test')
        val_dataset = datasets(root, 'Image_val')

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                                   num_workers=1)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    else:
        train_dataset = datasets(args['dataset_root'], 'Image_train')
        test_dataset = datasets(args['dataset_root'], 'Image_test')
        val_dataset = datasets(args['dataset_root'], 'Image_val')

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                                   num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    return train_loader, val_loader, test_loader


class pressure_map_datasets(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.img_names = os.listdir(root)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            Resize(IMG_SIZE),
            transforms.Pad(IMG_SIZE, fill=0),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_name = self.img_names[item]
        img_name_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_name_path, flags=3)[:, :, ::-1]
        img = self.transform(img)
        ret = re.match(r"[0-9]+_[0-9]+_([0-9]+.[0-9]+).jpg", img_name)
        bmi = float(ret.group(1))
        return (img,img_name), bmi


class datasets(data.Dataset):
    def __init__(self, root, file, sim=False):
        self.file = os.path.join(root, file)
        self.img_names = os.listdir(self.file)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            Resize(IMG_SIZE),
            transforms.Pad(IMG_SIZE, fill=0),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
        ])
        self.sim = sim

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_name_path = os.path.join(self.file, img_name)

        img = cv2.imread(img_name_path, flags=3)[:, :, ::-1]

        h, w, _ = img.shape

        img = self.transform(img)
        img = transforms.Normalize(IMG_MEAN, IMG_STD)(img)

        ret = re.match(r"[a-zA-Z0-9]+_[a-zA-Z0-9]+__?(\d+)__?(\d+)__?([a-z]+)_*", img_name)
        height = float(ret.group(2)) * 0.0254
        weight = float(ret.group(1)) * 0.4536
        sex = (lambda x: x == 'false')(ret.group(3))
        BMI = weight / (height ** 2)

        if self.sim:
            return img, BMI
        return (img, img_name_path, img_name), (sex, BMI)
