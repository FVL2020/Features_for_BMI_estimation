import json
import torch
import numpy as np
from pathlib import Path
import time
import os
from sklearn.metrics import mean_absolute_error
from torch.utils.tensorboard import SummaryWriter


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer(object):
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, DEVICE, optimizer, criterion, visual_path, save_dir=None, save_freq=50, mult_gpu=False):
        self.DEVICE = DEVICE
        self.model = model.to(self.DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_dir = save_dir
        self.best_error = 100
        self.save_freq = save_freq
        self.mult_gpu = mult_gpu
        self.writer = SummaryWriter('visual loss/' + visual_path)

    def _iteration(self, dataloader, epoch, mode='Test', sex='Overall'):
        epoch_time = AverageMeter('Time')
        losses = AverageMeter('Loss')
        error = AverageMeter('MAE')
        mape = AverageMeter('MAPE')

        if sex != 'Overall' and mode == 'Test' and sex is not None:
            male_error = AverageMeter('Male_MAE')
            male_mape = AverageMeter('Male_MAPE')
            female_error = AverageMeter('Female_MAE')
            female_mape = AverageMeter('Female_MAPE')

        t = time.time()

        for data, target in dataloader:
            data, _, _ = data
            sex, target = target
            data, target = data.to(self.DEVICE), target.to(self.DEVICE)
            self.optimizer.zero_grad()
            output = self.model(data)

            target = torch.unsqueeze(target, 1)
            bs = target.size(0)
            loss = self.criterion(output.double(), target.double())
            losses.update(loss.item(), bs)
            # print(output.shape)
            output_mae = output.detach().cpu().numpy()
            target_mae = target.detach().cpu().numpy()
            error_ = mean_absolute_error(target_mae, output_mae)
            error.update(error_)
            mape_ = mean_absolute_percentage_error(target_mae, output_mae)
            mape.update(mape_)

            # if mode == 'Train':
            self.writer.add_scalar(mode + ' Loss', loss.item(), epoch)
            self.writer.add_scalar(mode + ' MAE:', error_, epoch)
            self.writer.add_scalar(mode + ' MAPE:', mape_, epoch)


            if sex != 'Overall' and mode == 'Test' and sex is not None:
                if sex == 1:
                    male_error.update(error_)
                    male_mape.update(mape_)
                elif sex == 0:
                    female_error.update(error_)
                    female_mape.update(mape_)

            if mode == "Train":
                loss.backward()
                self.optimizer.step()

        epoch_time.update(time.time() - t)

        result = '\t'.join([
            '%s' % mode,
            'Time: %.3f' % epoch_time.val,
            'Loss: %.4f (%.4f)' % (losses.val, losses.avg),
            'MAE: %.4f (%.4f)' % (error.val, error.avg),
            'MAPE: %.4f (%.4f)' % (mape.val, mape.avg),
        ])
        print(result)
        if sex != 'Overall' and mode == 'Test' and sex is not None :
            dif_sex_result = '\t'.join([
                'Male_MAE: %.4f' % male_error.avg,
                'Male_MAPE: %.2f' % male_mape.avg,
                'Female_MAE: %.4f' % female_error.avg,
                'Female_MAPE: %.2f' % female_mape.avg,
            ])
            print(dif_sex_result)

        if mode == "Val":
            is_best = error.avg < self.best_error
            self.best_error = min(error.avg, self.best_error)
            if self.mult_gpu:
                MODEL = self.model.module
            else:
                MODEL = self.model
            if (is_best):
                self.save_checkpoint(state={
                    "epoch": epoch,
                    "state_dict": MODEL.state_dict(),
                    'MAE': self.best_error,
                    'MAPE': mape.avg,
                    'optimizer': self.optimizer.state_dict(),
                }, epoch=epoch, mode='best')
            if (epoch % self.save_freq) == 0:
                self.save_checkpoint(state={
                    "epoch": epoch,
                    "state_dict": MODEL.state_dict(),
                    'MAE': error.avg,
                    'MAPE': mape.avg,
                    'optimizer': self.optimizer.state_dict(),
                }, epoch=epoch, mode='normal')

        return mode, epoch_time.avg, losses.avg, error.avg, mape.avg

    def train(self, dataloader, epoch, mode='Train'):
        self.model.train()
        with torch.enable_grad():
            mode, t, loss, error, mape = self._iteration(dataloader, epoch=epoch, mode=mode)
            return mode, t, loss, error, mape

    def test(self, dataloader, epoch=None, mode='Test', sex='Overall'):
        self.model.eval()

        with torch.no_grad():
            mode, t, loss, error, mape = self._iteration(dataloader, epoch=epoch, mode=mode, sex=sex)
            if mode == 'Test':
                self.save_statistic(1, mode, t, loss, error, mape)
            return mode, t, loss, error, mape

    def Loop(self, epochs, trainloader, testloader, scheduler=None):
        for epoch in range(1, epochs + 1):

            print('Epoch: [%d/%d]' % (epoch, epochs))
            self.save_statistic(*((epoch,) + self.train(trainloader, epoch=epoch, mode='Train')))
            self.save_statistic(*((epoch,) + self.test(testloader, epoch=epoch, mode='Val')))
            print()
            if scheduler:
                scheduler.step()

    def save_checkpoint(self, state=None, epoch=0, mode='noraml', **kwargs):
        if self.save_dir:
            model_path = Path(self.save_dir)
            if not model_path.exists():
                model_path.mkdir()
            if mode == 'normal':
                torch.save(state, os.path.join(self.save_dir, "model_epoch_{}.ckpt".format(epoch)))
            elif mode == 'best':
                torch.save(state, os.path.join(self.save_dir, "best_model.ckpt"))

    def load(self, model_pth):
        checkpoint = torch.load(model_pth, map_location='cuda:1')
        error = checkpoint['MAE']
        mape = checkpoint['MAPE']
        epoch = checkpoint['epoch']
        pred_optimizer_dict = checkpoint['optimizer']
        optimizer_dict = self.optimizer.state_dict()
        pred_optimizer_dict = {k: v for k, v in pred_optimizer_dict.items() if k in optimizer_dict}
        optimizer_dict.update(pred_optimizer_dict)

        model_dict = self.model.state_dict()
        pred_dict = checkpoint['state_dict']
        pred_dict = {k: v for k, v in pred_dict.items() if k in model_dict \
                     and (k != 'fc2.1.weight')}
        model_dict.update(pred_dict)
        # self.optimizer.load_state_dict(optimizer_dict)
        self.model.load_state_dict(model_dict)

        # 冻结卷积层，只训练全连接层
        # for para in self.model.parameters():
        #     para.requires_grad = False
        # for para in self.model.fc1.parameters():
        #     para.requires_grad = True
        # for para in self.model.fc2.parameters():
        #     para.requires_grad = True

        print('The %d epoch model performed val MAE: %f\t MAPE: %f' % (epoch, error, mape))
        print('optimizer：')
        for var_name in optimizer_dict['param_groups'][0]:
            if var_name != 'params':
                print(var_name, "\t", optimizer_dict['param_groups'][0][var_name])

    def save_statistic(self, epoch, mode, t, loss, error, mape):
        if self.save_dir:
            model_path = Path(self.save_dir)
            if not model_path.exists():
                model_path.mkdir()
        with open(self.save_dir + '/state.txt', 'a', encoding='utf-8') as f:
            f.write(str({"epoch": epoch, "mode": mode, "time": t, "loss": loss, "MAE": error, "MAPE": mape}))
            f.write('\n')
