from sklearn.metrics import f1_score, precision_score, recall_score
# , classification_report, confusion_matrix
from model_harmonic import HarmonicNet
from loader import Loader
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import gc
from typing import List


lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


metadata = pd.read_csv('HAM10000_metadata.csv')


enc = LabelEncoder()
metadata['dx'] = enc.fit_transform(metadata['dx'])
metadata['dx_type'] = enc.fit_transform(metadata['dx_type'])
metadata['sex'] = enc.fit_transform(metadata['sex'])
metadata['localization'] = enc.fit_transform(metadata['localization'])
metadata['lesion_id'] = enc.fit_transform(metadata['lesion_id'])
labels = metadata.dx.values

imageid_path_dict = {x: f'HAM10000_images/{x}.jpg' for x in metadata.image_id}

trainset = Loader(imageid_path_dict, labels, train=True, transform=None)
testset = Loader(imageid_path_dict, labels, train=False, transform=None)
train_sampler = torch.utils\
    .data.WeightedRandomSampler(trainset.weights[trainset.train_labels],
                                len(
        trainset.weights[trainset.train_labels]),
        True)
test_sampler = torch.utils\
    .data.WeightedRandomSampler(testset.weights[testset.test_labels],
                                len(
        testset.weights[testset.test_labels]),
        True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=50,
                                          sampler=train_sampler,
                                          shuffle=False,
                                          num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         sampler=test_sampler,
                                         shuffle=False,
                                         num_workers=2)
gc.collect()

model_harmonic = HarmonicNet(3, 32, 4, 1, 1, (32, 32))


for module in model_harmonic.modules():
    if isinstance(module, nn.Conv2d):
        module.weight.data.normal_(0, 0.05)
        module.bias.data.zero_()

model_harmonic = model_harmonic.to('cuda')
for parameter in model_harmonic.parameters():
    parameter.to('cuda')
model_harmonic.cuda()
model_harmonic = torch.nn.DataParallel(
    model_harmonic, device_ids=range(torch.cuda.device_count()))
base_lr = 0.01
param_dict = dict(model_harmonic.named_parameters())
params = []  # type: List
train_losses = []  # type: List[float]
test_losses = []  # type: List[float]
train_accs = []  # type: List[float]
test_accs = []  # type: List[float]

criterion = nn.CrossEntropyLoss()
# BCEWithLogitsLoss()
# MultiLabelSoftMarginLoss()
# BCELoss with tanh, atan
# MultiLabelSoftMarginLoss()
# BCEWithLogitsLoss()

optimizer = optim.Adam(model_harmonic.parameters(),
                       # eps=1e-5,
                       lr=base_lr,
                       weight_decay=0.)

best_acc = 0

gc.collect()


trainloader = torch.utils.data.DataLoader(trainloader,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=2)
testloader = torch.utils.data.DataLoader(testloader,
                                         batch_size=64,
                                         shuffle=True,
                                         num_workers=2)
classes = list(lesion_type_dict.keys())


def one_hot_enc(output, target, num_classes=7):
    labels = target.view((-1, 1))
    batch_size, _ = labels.size()
    labels_one_hot = torch.FloatTensor(
        batch_size, num_classes).zero_().to('cuda')
    labels_one_hot.scatter_(1, labels, 1)
    return labels_one_hot


def train(epoch, model):
    model.train()
    # w = trainloader.dataset.weights
    corrects = 0.0
    f1 = 0.0
    prec = 0.0
    rec = 0.0
    iteration = 0
    # criterion.weight = torch.from_numpy(trainset.weights)\
    # .to('cuda').type(torch.float)
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = torch.autograd.Variable(
            data.cuda()), torch.autograd.Variable(label.cuda())
        optimizer.zero_grad()
        output = model(data)
        label_ = label  # one_hot_enc(output, label)
        loss = criterion(output, label_)
        y_pred = torch.max(output, 1)[1]
        loss.backward()
        optimizer.step()
        corrects += y_pred.eq(label.data).cpu().sum()
        f1 += f1_score(y_true=label.data.cpu().numpy(),
                       y_pred=y_pred.cpu().numpy(),
                       average='micro')
        prec += precision_score(y_true=label.data.cpu().numpy(),
                                y_pred=y_pred.cpu().numpy(),
                                average='micro')
        rec += recall_score(y_true=label.data.cpu().numpy(),
                            y_pred=y_pred.cpu().numpy(),
                            average='micro')
        iteration += 1
    acc = 100. * corrects / len(trainloader.dataset)
    f1 = f1/iteration
    prec = prec/iteration
    rec = rec/iteration
    train_losses.append(loss.data.item())
    train_accs.append(acc)
    print(
        f"\tTraining accuracy = {acc:.2f}%; F1 = {100.*f1:.2f}%;\
            Precision = {100.*prec:.2f}%; Recall = {100.*rec:.2f}%\n")


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
    for batch_id, (data, label) in enumerate(testloader):
        with torch.no_grad():
            data, label = torch.autograd.Variable(
                data).cuda(), torch.autograd.Variable(label).cuda()
            output = model(data)
            # one_hot_enc(output, label) # one hot encoding for loss function
            label_ = label
            loss = criterion(output, label_)
            y_pred = torch.max(output, 1)[1]
            corrects += y_pred.eq(label.data).cpu().sum()
            testloss += loss
            f1 += f1_score(y_true=label.data.cpu().numpy(),
                           y_pred=y_pred.cpu().numpy(),
                           average='micro')
            prec += precision_score(y_true=label.data.cpu().numpy(),
                                    y_pred=y_pred.cpu().numpy(),
                                    average='micro')
            rec += recall_score(y_true=label.data.cpu().numpy(),
                                y_pred=y_pred.cpu().numpy(),
                                average='micro')
        iteration += 1
    acc = 100. * corrects / len(testloader.dataset)
    f1 = f1/iteration
    prec = prec/iteration
    rec = rec/iteration
    testloss /= len(testloader.dataset)
    if best_acc < acc.item():
        best_acc = acc
        save_state(model, best_acc)
    print(
        f"\tTesting accuracy = {acc:.2f}%; F1 = {100.*f1:.2f}%; \
            Precision = {100.*prec:.2f}%;\
            Recall = {100.*rec:.2f}% \n\tLoss: {testloss:1.2e}\n")
    test_losses.append(loss.data.item())
    test_accs.append(acc)


def adjust_learning_rate(optimizer, epoch):
    update_list = [i for i in range(0, 150, 10)]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * 0.1, 1e-8)
    return


def save_state(model, best_acc):
    print('\n==> Saving model ...')
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
