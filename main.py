import os
import time
from sklearn.metrics import f1_score, precision_score, recall_score
# , classification_report, confusion_matrix
# from torchvision.transforms import ToTensor
from models.model_harmonic import WideHarmonicResNet
# from local_loader import LocalLoader
# from loader import LoaderSmall
# import pandas as pd
import warnings
import torch
from torchvision.datasets import CIFAR10
from torch import nn
import torch.optim as optim
# from sklearn.preprocessing import LabelEncoder
import gc
from typing import List  # pylint: ignore
from tqdm import tqdm
# import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import random
warnings.filterwarnings('ignore')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

trainset = CIFAR10('./', download=True, train=True,
                   transform=transform_train)  # LoaderSmall(imageid_path_dict, labels, train=True, transform=tfms, color_space=None)
# LoaderSmall(imageid_path_dict, labels, train=True, transform=tfms, color_space=None)
# #CIFAR10('./', download=True, train=True, transform=ToTensor())#
testset = CIFAR10('./',
                  train=False,
                  transform=transform_test)  # LoaderSmall(imageid_path_dict, labels, train=False, transform=tfms, color_space=None)
# LoaderSmall(imageid_path_dict, labels, train=False, transform=tfms, color_space=None)
# CIFAR10('./', train=False, transform=ToTensor())#
'''
train_sampler = torch.utils\
    .data.WeightedRandomSampler(trainset.weights[trainset.train_labels],
                                len(trainset.weights[trainset.train_labels]),
                                True)
test_sampler = torch.utils\
    .data.WeightedRandomSampler(testset.weights[testset.test_labels],
                                len(testset.weights[testset.test_labels]),
                                True)
'''
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          # sampler=train_sampler,
                                          shuffle=True,
                                          num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         # sampler=test_sampler,
                                         shuffle=False,
                                         num_workers=0)
gc.collect()

net = WideHarmonicResNet(3, 28, widen_factor=3)

for module in net.modules():
    if isinstance(module, nn.Conv2d):
        module.weight.data.kaiming_normal_(0, 0.05)
        if module.bias is not None:
            module.bias.data.zero_()

net = net.to('cuda')
for parameter in net.parameters():
    parameter.to('cuda')
net.cuda()
net = torch.nn.DataParallel(
    net, device_ids=range(torch.cuda.device_count()))
base_lr = 0.1
param_dict = dict(net.named_parameters())
params = []  # type: List
train_losses = []  # type: List[float]
test_losses = []  # type: List[float]
train_accs = []  # type: List[float]
test_accs = []  # type: List[float]
train_precisions = []  # type: List[float]
test_precisions = []  # type: List[float]
train_f1 = []  # type: List[float]
test_f1 = []  # type: List[float]
train_recall = []  # type: List[float]
test_recall = []  # type: List[float]

criterion = nn.CrossEntropyLoss()  # BCEWithLogitsLoss()
# BCEWithLogitsLoss()

optimizer = optim.SGD(params=net.parameters(),
                      lr=base_lr,
                      momentum=0.9,
                      dampening=0,
                      weight_decay=0.0005)  # Adam(model_harmonic.parameters(),
#     lr=base_lr,
#     weight_decay=1e-3)

best_acc = 0

gc.collect()


def one_hot_enc(output, target, num_classes=7):
    labels = target.view((-1, 1))
    batch_size, _ = labels.size()
    labels_one_hot = torch.FloatTensor(
        batch_size, num_classes).zero_().to('cuda')
    labels_one_hot.scatter_(1, labels, 1)
    return labels_one_hot


# Training (https://github.com/kuangliu/pytorch-cifar/blob/master/main.py)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = net(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(
        'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (len(trainloader)), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            test_loss / (len(testloader)), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(f'checkpoint_cifar10_{time.strftime("%Y_%m_%d")}'):
            os.mkdir(f'checkpoint_cifar10_{time.strftime("%Y_%m_%d")}')
        torch.save(state,
                   f'./checkpoint_cifar10_{time.strftime("%Y_%m_%d")}/ckpt_{time.strftime("%H_%M_%S")}_{acc:.2f}.t7')
        best_acc = acc


def adjust_learning_rate(optimizer,
                         epoch,
                         update_list=(25, 75),
                         factor=10.,
                         lim=1.):
    # [60, 120, 160]  #[2,5,8,11,14,17,20]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * factor, lim)
    return


def save_state(model, best_acc):
    print('\n==> Saving model ...\n')
    state = {'best_acc': best_acc,
             'state_dict': model.state_dict()}
    keys = list(state['state_dict'].keys())
    for key in keys:
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                state['state_dict'].pop(key)
    torch.save(state, 'harmonic_network.tar')


def get_lr(optimizer=optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# t = tqdm(total=300)
for epoch in range(200):
    adjust_learning_rate(optimizer, epoch, [60, 120, 160], factor=0.2, lim=1e-6)
    lr = get_lr()
    print(f"Epoch: {epoch}, learning rate = {lr:1.1e};")
    train(epoch)
    test(epoch)
    gc.collect()
    torch.cuda.empty_cache()
