import torchvision.models as models
import torch.nn as nn
import torch

df = 15


def Resnet50(df=df, num_class=1, ):
    net = models.resnet50(pretrained=True)
    net.fc = nn.Sequential(
        nn.Linear(2048, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def VGG16(df=df, num_class=1, ):
    net = models.vgg16()
    net.classifier = nn.Sequential(
        nn.Linear(7 * 7 * 512, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def Mobilenetv2(df, num_class=1, ):
    net = models.mobilenet_v2()
    net.classifier = nn.Sequential(
        nn.Linear(1280, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def Densenet121(df=df, num_class=1, ):
    net = models.densenet121()
    net.classifier = nn.Sequential(
        nn.Linear(1024, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def Resnext50(df=df, num_class=1, ):
    net = models.resnext50_32x4d()
    net.fc = nn.Sequential(
        nn.Linear(2048, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def GhostNet(df=df, num_class=1):
    net = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
    net.classifier = nn.Sequential(
        nn.Linear(1280, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def Resnest50(df=df, num_class=1):
    net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    net.fc = nn.Sequential(
        nn.Linear(2048, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


#
def GoogleNetv3(df=df, num_class=1):
    net = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
    net.fc = nn.Sequential(
        nn.Linear(2048, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net

if __name__ == '__main__':
    print(Resnest50(15))
