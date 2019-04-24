from sklearn.metrics import f1_score, precision_score, recall_score
# , classification_report, confusion_matrix
from torchvision.transforms import ToTensor
from models.model_harmonic import WideHarmonicResNet
# from local_loader import LocalLoader
from loader import LoaderSmall
import pandas as pd
import warnings
import torch
from torchvision.datasets import CIFAR10
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import gc
from typing import List  # pylint: ignore
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

warnings.filterwarnings('ignore')

torch.random.manual_seed(42)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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

net = WideHarmonicResNet(3, 28, widen_factor=10)

for module in net.modules():
    if isinstance(module, nn.Conv2d):
        module.weight.data.normal_(0, 0.05)
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


def train(epoch, model):
    model.train()
    corrects = 0.0
    f1 = 0.0
    prec = 0.0
    rec = 0.0
    iteration = 0
    acc = 0.
    for batch_idx, (data, label) in enumerate(tqdm(trainloader)):
        data, label = torch.autograd.Variable(
            data.cuda()), torch.autograd.Variable(label.cuda())
        optimizer.zero_grad()
        output = model(data)
        # label_ = one_hot_enc(output, label, 10)
        loss = criterion(output, label)
        y_pred = torch.max(output, 1)[1]
        loss.backward()
        optimizer.step()
        corrects += y_pred.eq(label.data).cpu().sum()
        f1 += f1_score(y_true=label.data.cpu().numpy(),
                       y_pred=y_pred.cpu().numpy(),
                       average='weighted')
        prec += precision_score(y_true=label.data.cpu().numpy(),
                                y_pred=y_pred.cpu().numpy(),
                                average='weighted')
        rec += recall_score(y_true=label.data.cpu().numpy(),
                            y_pred=y_pred.cpu().numpy(),
                            average='weighted')
        iteration += 1
        # t.update(batch_idx)
    acc = 100. * corrects / len(trainloader.dataset)
    f1 = f1 / iteration
    prec = prec / iteration
    rec = rec / iteration
    train_losses.append(loss.data.item())
    train_accs.append(acc)
    train_precisions.append(100. * rec)
    train_recall.append(100. * rec)
    train_f1.append(100. * f1)
    print(f"\nTraining accuracy = {acc:.2f}%;\n\
             F1 = {100. * f1:.2f}%;\
             Precision = {100. * prec:.2f}%;\
             Recall = {100. * rec:.2f}%\
             Loss: {loss.data.item():1.2e}\n")

    # t.close()


def test(epoch, model):
    global best_acc
    model.eval()
    # w = testloader.dataset.weights
    acc = 0.0
    f1 = 0.0
    prec = 0.0
    rec = 0.0
    iteration = 0
    testloss = 0.0
    corrects = 0.0
    # criterion.weight = torch.from_numpy(testset.weights)\
    # .to('cuda').type(torch.float)
    # t = tqdm(total=len(testloader))
    for batch_id, (data, label) in enumerate(tqdm(testloader)):
        with torch.no_grad():
            data, label = torch.autograd.Variable(
                data).cuda(), torch.autograd.Variable(label).cuda()
            output = model(data)
            # one_hot_enc(output, label) # one hot encoding for loss function
            # label_ = one_hot_enc(output, label, 10)
            loss = criterion(output, label)
            y_pred = torch.max(output, 1)[1]
            corrects += y_pred.eq(label.data).cpu().sum()
            testloss += loss
            f1 += f1_score(y_true=label.data.cpu().numpy(),
                           y_pred=y_pred.cpu().numpy(),
                           average='weighted')
            prec += precision_score(y_true=label.data.cpu().numpy(),
                                    y_pred=y_pred.cpu().numpy(),
                                    average='weighted')
            rec += recall_score(y_true=label.data.cpu().numpy(),
                                y_pred=y_pred.cpu().numpy(),
                                average='weighted')
        iteration += 1
        # t.update(batch_id)
    acc = 100. * corrects / len(testloader.dataset)
    f1 = f1 / iteration
    prec = prec / iteration
    rec = rec / iteration
    testloss /= len(testloader.dataset)
    if best_acc < acc.item():
        best_acc = acc
        save_state(model, best_acc)
    print(
        f"\nTesting accuracy = {acc:.2f}%; \n \
            F1 = {100. * f1:.2f}%; \
            Precision = {100. * prec:.2f}%;\
            Recall = {100. * rec:.2f}% \
            Loss: {loss.data.item():1.2e}\n")
    test_losses.append(loss.data.item())
    test_accs.append(acc)
    test_precisions.append(100. * rec)
    test_recall.append(100. * rec)
    test_f1.append(100. * f1)
    # t.close()


def adjust_learning_rate(optimizer,
                         epoch,
                         update_list=[25, 75],
                         factor=10,
                         lim=1.):
    # [60, 120, 160]  #[2,5,8,11,14,17,20]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = min(param_group['lr'] * factor, lim)
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
    adjust_learning_rate(optimizer, epoch, [60, 120, 160], factor=0.2, lim=1e-4)
    lr = get_lr()
    print(f" Epoch: {epoch}, learning rate = {lr:1.1e};\n")
    train(epoch, net)
    test(epoch, net)
    gc.collect()
    if np.nan in test_losses or np.nan in train_losses:
        break
    torch.cuda.empty_cache()
