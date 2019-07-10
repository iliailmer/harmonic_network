import os
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from models.net import WideOrthoResNet, OrthoVGG
from layers.blocks import BasicBlock, HadamardBlock, HarmonicBlock, SlantBlock
from loader import LoaderSmall, Loader
import pandas as pd
import warnings
import torch
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gc
from typing import List  # pylint: ignore
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
import random
import argparse
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('AGG')

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--block', type=str, default='basic')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--alpha-root', type=float, default=None)
parser.add_argument('--depth', type=int, default=16)
parser.add_argument('--widen', type=int, default=1)
parser.add_argument('--lam', type=int, default=None)
parser.add_argument('--diag', type=bool, default=False)
parser.add_argument('--bs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--arch', type=str, default='wrn')
parser.add_argument('--cfg', type=str, default='A')
parser.add_argument('--imsize', type=str, default='small')  # for skin dataset only
parser.add_argument('--checkpoint', type=int, default=0)
args = parser.parse_args()

folder = {'small': 'HAM10000_small',
          'large': 'HAM10000_224',
          'isic':'ISIC2019/ISIC_2019_Training_Input'}
assert args.imsize in list(folder.keys()), "Unexpected folder name, expected one of {}, got {}".format(
    list(folder.keys()),
    args.imsize)

blocks = {'basic': BasicBlock,
          'hadamard': HadamardBlock,
          'harmonic': HarmonicBlock,
          'slant': SlantBlock}
assert args.block in list(blocks.keys()), "Unexpected block name, expected one of {}, got {}".format(
    list(blocks.keys()),
    args.block)

archs = {'wrn': WideOrthoResNet,
         'vgg': OrthoVGG}
assert args.arch in list(archs.keys()), "Unexpected acrh name, expected one of {}, got {}".format(list(archs.keys()),
                                                                                                  args.arch)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

weights = None

if args.dataset not in ['cifar10', 'cifar100', 'imagenet', 'isic2019']:
    use_cm=True
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    '''transforms.ToTensor(),
    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                      (4, 4, 4, 4),
                                      mode='reflect').squeeze()),
    transforms.ToPILImage(),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),'''
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    print("Loading metadata...")
    metadata = pd.read_csv('metadata/HAM10000_metadata.csv')
    enc = LabelEncoder()
    metadata['dx'] = enc.fit_transform(metadata['dx'])
    metadata['dx_type'] = enc.fit_transform(metadata['dx_type'])
    metadata['sex'] = enc.fit_transform(metadata['sex'])
    metadata['localization'] = enc.fit_transform(metadata['localization'])
    metadata['lesion_id'] = enc.fit_transform(metadata['lesion_id'])
    labels = metadata.dx.values
    imageid_path_dict = {x: f'./{folder[args.imsize]}/{x}.jpg' for x in metadata.image_id}

    print("Loading Images...")
    train_names, val_names, \
    train_labels, val_labels = train_test_split(
        np.asarray(list(imageid_path_dict.keys())),
        labels,
        test_size=0.15)

    '''weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(labels),
        labels)'''
    trainset = LoaderSmall(imageid_path_dict,
                           labels=train_labels,
                           names=train_names,
                           weighting=False,
                           transform=transform_train,
                           color_space=None)

    testset = LoaderSmall(imageid_path_dict,
                          labels=val_labels,
                          names=val_names,
                          weighting=False,
                          transform=transform_test,
                          color_space=None)
    #get train sampler
    target = torch.from_numpy(train_labels)
    class_sample_count = torch.tensor(
        [(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in target])
    train_sampler = torch.utils \
        .data.WeightedRandomSampler(samples_weight,
                                    len(samples_weight))
    #get test sampler
    target = torch.from_numpy(val_labels)
    class_sample_count = torch.tensor(
        [(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in target])
    test_sampler = torch.utils \
        .data.WeightedRandomSampler(samples_weight,
                                    len(samples_weight))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs,
                                              sampler=train_sampler,
                                              num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs,
                                             sampler=test_sampler,
                                             num_workers=2)

    target = torch.from_numpy(labels)
    class_sample_count = torch.tensor(
        [(target == t).sum() for t in torch.unique(target, sorted=True)])
    weights = 1. / class_sample_count.float()

    if args.arch == 'wrn':
        net = WideOrthoResNet(in_channels=3,
                              block=blocks[args.block],
                              alpha_root=args.alpha_root,
                              kernel_size=args.kernel_size,
                              depth=args.depth,
                              num_classes=7,
                              widen_factor=args.widen,
                              lmbda=args.lam,
                              diag=args.diag)
    if args.arch == 'vgg':
        net = OrthoVGG(cfg=args.cfg,
                       block=blocks[args.block],
                       in_channels=3,
                       num_classes=7,
                       lmbda=args.lam,
                       diag=args.diag)

elif args.dataset == 'isic2019':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          [4, 4, 4, 4],
                                          mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(224, padding=4),
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
    print("Loading metadata...")
    meta = pd.read_csv('ISIC2019/ISIC_2019_Training_GroundTruth.csv')
    labels = {}
    for x in meta.values:
        labels[x[0]] = np.argmax(x[1:].astype(np.int))

    imageid_path_dict = {x: f'./ISIC2019/ISIC_2019_Training_Input/{x}.jpg' for x in list(labels.keys())}
    labels = np.array(list(labels.values()))
    train_names, val_names, \
    train_labels, val_labels = train_test_split(
        np.asarray(list(imageid_path_dict.keys())),
        labels,
        test_size=0.20)
    print("Loading Images...")
    trainset = Loader(imageid_path_dict,
                           names=train_names,
                           labels=train_labels,
                           transform=transform_train)
    testset = Loader(imageid_path_dict,
                          names=val_names,
                          labels=val_labels,
                          transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs,
                                              shuffle=False,
                                              num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                             shuffle=False,
                                             num_workers=2)
    if args.arch == 'wrn':
        net = WideOrthoResNet(in_channels=3,
                              block=blocks[args.block],
                              alpha_root=args.alpha_root,
                              kernel_size=args.kernel_size,
                              depth=args.depth,
                              num_classes=8,
                              droprate=0.2,
                              widen_factor=args.widen,
                              lmbda=args.lam,
                              diag=args.diag)
    if args.arch == 'vgg':
        net = OrthoVGG(cfg=args.cfg,
                       block=blocks[args.block],
                       in_channels=3,
                       num_classes=8,
                       lmbda=args.lam,
                       diag=args.diag)

elif args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          [4, 4, 4, 4],
                                          mode='reflect').squeeze()),
        transforms.ToPILImage(),
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
                       transform=transform_train)
    testset = CIFAR10('./',
                      train=False,
                      transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs,
                                              shuffle=True,
                                              num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False,
                                             num_workers=0)
    if args.arch == 'wrn':
        net = WideOrthoResNet(in_channels=3,
                              block=blocks[args.block],
                              alpha_root=args.alpha_root,
                              kernel_size=args.kernel_size,
                              depth=args.depth,
                              widen_factor=args.widen,
                              num_classes=10,
                              lmbda=args.lam,
                              diag=args.diag)
    if args.arch == 'vgg':
        net = OrthoVGG(cfg=args.cfg,
                       block=blocks[args.block],
                       in_channels=3,
                       num_classes=10,
                       lmbda=args.lam,
                       diag=args.diag)

elif args.dataset == 'cifar100':
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
    trainset = CIFAR100('./', download=True, train=True,
                        transform=transform_train)
    testset = CIFAR100('./',
                       train=False,
                       transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True,
                                              num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False,
                                             num_workers=0)
    if args.arch == 'wrn':
        net = WideOrthoResNet(in_channels=3,
                              block=blocks[args.block],
                              alpha_root=None,
                              kernel_size=args.kernel_size,
                              depth=args.depth,
                              num_classes=100,
                              widen_factor=args.widen,
                              lmbda=args.lam,
                              diag=args.diag)

    if args.arch == 'vgg':
        net = OrthoVGG(cfg=args.cfg,
                       block=blocks[args.block],
                       in_channels=3,
                       num_classes=10,
                       lmbda=args.lam,
                       diag=args.diag)
gc.collect()

print("Number of trainable parameters:", net._num_parameters()[0])
if args.block != 'basic':
    print("Alpha-root:", net.conv1.alpha_root)
    print("Requires grad:", net.conv1.filter_bank.requires_grad)

net = net.cuda()
net = torch.nn.DataParallel(
    net,
    device_ids=list(range(torch.cuda.device_count())))
base_lr = args.lr
best_acc = 0
start_epoch = 0
assert args.checkpoint in [0, 1], f"Checkpoint must be 1 or 0, got {args.checkpoint}"
if args.checkpoint == 1:
    model_dict = torch.load(f'./checkpoint_{args.dataset}_2019_06_20/' +
                            f'ckpt_slant_wrn_22_3_4x4__94.48.t7')
    start_epoch = model_dict['epoch']
    net.load_state_dict(model_dict['net'])
    best_acc = model_dict['acc']
    base_lr = model_dict['lr']

param_dict = dict(net.named_parameters())
params = []  # type: List
train_losses = []  # type: List[float]
test_losses = []  # type: List[float]
train_accs = []  # type: List[float]
test_accs = []  # type: List[float]
train_error = []
test_error = []

if weights is not None:
    criterion = nn.CrossEntropyLoss(weight=weights.cuda())  # weight=torch.from_numpy(weights).cuda())
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(),
                      lr=base_lr,
                      momentum=0.9,
                      dampening=0,
                      weight_decay=args.weight_decay,
                      nesterov=True)

gc.collect()


def get_lr(optimizer=optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def one_hot_enc(output, target, num_classes=7):
    labels = target.view((-1, 1))
    batch_size, _ = labels.size()
    labels_one_hot = torch.FloatTensor(batch_size, num_classes).zero_().to('cuda')
    labels_one_hot.scatter_(1, labels, 1)
    return labels_one_hot


# Training (https://github.com/kuangliu/pytorch-cifar/blob/master/main.py)
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    print(f"\nEpoch: {epoch}, learning rate = {lr:1.1e};")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    f1 = 0
    prec = 0
    rec = 0
    acc = 0
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
        f1 += f1_score(y_pred=predicted.cpu().numpy(),
                       y_true=targets.cpu().numpy(),
                       average="weighted")
        prec += precision_score(y_pred=predicted.cpu().numpy(),
                                y_true=targets.cpu().numpy(),
                                average="weighted")
        rec += recall_score(y_pred=predicted.cpu().numpy(),
                            y_true=targets.cpu().numpy(),
                            average="weighted")
        acc += accuracy_score(y_pred=predicted.cpu().numpy(),
                              y_true=targets.cpu().numpy())
    print(
        'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (len(trainloader)), 100. * correct / total, correct, total))
    print(
        'Precision: %.3f | F1: %.3f | Recall: %.3f | Acc (sklearn): %.3f%%\n' % \
        (prec/ len(trainloader), f1/len(trainloader), rec/ len(trainloader), 100.*acc / len(trainloader)))
    train_accs.append(100. * correct / total)
    train_error.append(100. - 100. * correct / total)
    train_losses.append(train_loss)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    f1 = 0
    prec = 0
    rec = 0
    acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            f1 += f1_score(y_pred=predicted.cpu().numpy(),
                           y_true=targets.cpu().numpy(),
                           average="weighted")
            prec += precision_score(y_pred=predicted.cpu().numpy(),
                                    y_true=targets.cpu().numpy(),
                                    average="weighted")
            rec += recall_score(y_pred=predicted.cpu().numpy(),
                                y_true=targets.cpu().numpy(),
                                average="weighted")
            acc += accuracy_score(y_pred=predicted.cpu().numpy(),
                                  y_true=targets.cpu().numpy())
        print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Error: %.3f%%' % (
            test_loss / (len(testloader)), 100. * correct / total, correct, total, 100. * (1. - correct / total)))
        print(
            'Precision: %.3f | F1: %.3f | Recall: %.3f | Acc (sklearn): %.3f%%\n' % \
            (prec / len(testloader), f1 / len(testloader), rec / len(testloader), 100.*acc / len(testloader)))

        test_accs.append(100. * correct / total)
        test_error.append(100. - 100. * correct / total)
        test_losses.append(test_losses)
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'lr': get_lr(optimizer),
            # 'optimizer': optimizer.state_dict()

        }
        if not os.path.isdir(f'checkpoint_{args.dataset}_{time.strftime("%Y_%m_%d")}'):
            os.mkdir(f'checkpoint_{args.dataset}_{time.strftime("%Y_%m_%d")}')
        a = ''
        if args.alpha_root is not None:
            a = f'alpha_{args.alpha_root}_'

        torch.save(state,
                   f'./checkpoint_{args.dataset}_{time.strftime("%Y_%m_%d")}' +
                   f'/ckpt_{args.block}_{args.arch}_{args.depth}_{args.widen}_{args.kernel_size}x{args.kernel_size}_' +
                   a + f'_{acc:.2f}.t7')
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


for epoch in range(start_epoch, 200):
    adjust_learning_rate(optimizer, epoch, [60, 120, 160], factor=0.2, lim=1e-6)
    lr = get_lr()
    train(epoch)
    test(epoch)
    print(f"Best Accuracy: {best_acc:.3f}")
    gc.collect()
    torch.cuda.empty_cache()

if use_cm:
    confusion_matrix(y)
del trainset, testset, trainloader, testloader
gc.collect()

a = ''
if args.alpha_root is not None:
    a = f'alpha_{args.alpha_root}_'

'''fig = plt.figure()
ax = fig.add_subplot(111)
#plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
ax.plot(test_losses, label="Test Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
fig.savefig(f'./checkpoint_{args.dataset}_{time.strftime("%Y_%m_%d")}/losses_'+
            f'{args.block}_{args.arch}_{args.depth}_{args.widen}_'+
            a+f'{acc:.2f}.png')
plt.close()
gc.collect()'''

plt.style.use('ggplot')
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot()
ax.set_title(f"Block{args.block},\n{args.arch}-{args.depth}-{args.widen}, alpha-rooting={args.alpha_root}")
ax.plot(test_accs, label="Test Accuracy")
ax.plot(train_accs, label="Train Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
fig.savefig(f'./checkpoint_{args.dataset}_{time.strftime("%Y_%m_%d")}/accs_test_' +
            f'{args.block}_{args.arch}_{args.depth}_{args.widen}' +
            a + f'__{args.kernel_size}x{args.kernel_size}_{best_acc:.2f}.png')
plt.close()
np.save(f'./checkpoint_{args.dataset}_{time.strftime("%Y_%m_%d")}/accs_train_' +
        f'{args.block}_{args.arch}_{args.depth}_{args.widen}' +
        a + f'__{best_acc:.2f}', np.asarray(train_accs))
np.save(f'./checkpoint_{args.dataset}_{time.strftime("%Y_%m_%d")}/accs_test_' +
        f'{args.block}_{args.arch}_{args.depth}_{args.widen}' +
        a + f'__{args.kernel_size}x{args.kernel_size}_{best_acc:.2f}', np.asarray(test_accs))
gc.collect()
