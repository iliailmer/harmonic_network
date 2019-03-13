from sklearn.metrics import f1_score, precision_score, recall_score
# , classification_report, confusion_matrix
from model_harmonic import HarmonicNet
# from local_loader import LocalLoader
from loader import LoaderSmall
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import gc
from typing import List  # pylint: ignore
from tqdm import tqdm
import matplotlib.pyplot as plt
from albumentations.augmentations.transforms import RandomRotate90, Rotate
from albumentations import Compose


tfms = None  # Compose(transforms=[RandomRotate90(p=0.4), Rotate(p=0.5)])

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


metadata = pd.read_csv('metadata/HAM10000_metadata.csv')


enc = LabelEncoder()
metadata['dx'] = enc.fit_transform(metadata['dx'])
metadata['dx_type'] = enc.fit_transform(metadata['dx_type'])
metadata['sex'] = enc.fit_transform(metadata['sex'])
metadata['localization'] = enc.fit_transform(metadata['localization'])
metadata['lesion_id'] = enc.fit_transform(metadata['lesion_id'])
labels = metadata.dx.values

imageid_path_dict = {x: f'HAM10000_small/{x}.jpg' for x in metadata.image_id}
print("Loading data...\n")
trainset = LoaderSmall(imageid_path_dict, labels, train=True, transform=tfms)
testset = LoaderSmall(imageid_path_dict, labels, train=False, transform=tfms)

train_sampler = torch.utils\
    .data.WeightedRandomSampler(trainset.weights[trainset.train_labels],
                                len(trainset.weights[trainset.train_labels]),
                                True)
test_sampler = torch.utils\
    .data.WeightedRandomSampler(testset.weights[testset.test_labels],
                                len(testset.weights[testset.test_labels]),
                                True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          sampler=train_sampler,
                                          shuffle=False,
                                          num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         sampler=test_sampler,
                                         shuffle=False,
                                         num_workers=1)
gc.collect()

model_harmonic = HarmonicNet(3, 32, 4, 1, 1, (64, 64))

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
base_lr = 0.001
param_dict = dict(model_harmonic.named_parameters())
params = []  # type: List
train_losses = []  # type: List[float]
test_losses = []  # type: List[float]
train_accs = []  # type: List[float]
test_accs = []  # type: List[float]

criterion = nn.BCEWithLogitsLoss()
# BCEWithLogitsLoss()
# MultiLabelSoftMarginLoss()
# BCELoss with tanh, atan
# MultiLabelSoftMarginLoss()
# BCEWithLogitsLoss()

optimizer = optim.Adam(model_harmonic.parameters(),
                       # eps=1e-5,
                       lr=base_lr,
                       weight_decay=0.0001)

best_acc = 0

gc.collect()

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
    # t = tqdm(total=len(trainloader))
    for batch_idx, (data, label) in enumerate(tqdm(trainloader)):
        data, label = torch.autograd.Variable(
            data.cuda()), torch.autograd.Variable(label.cuda())
        optimizer.zero_grad()
        output = model(data)
        label_ = one_hot_enc(output, label)
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
        # t.update(batch_idx)
    acc = 100. * corrects / len(trainloader.dataset)
    f1 = f1/iteration
    prec = prec/iteration
    rec = rec/iteration
    train_losses.append(loss.data.item())
    train_accs.append(acc)
    print(
        f"\nTraining accuracy = {acc:.2f}%; F1 = {100.*f1:.2f}%;\
            Precision = {100.*prec:.2f}%; Recall = {100.*rec:.2f}%\n")
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
            label_ = one_hot_enc(output, label)
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
        # t.update(batch_id)
    acc = 100. * corrects / len(testloader.dataset)
    f1 = f1/iteration
    prec = prec/iteration
    rec = rec/iteration
    testloss /= len(testloader.dataset)
    if best_acc < acc.item():
        best_acc = acc
        save_state(model, best_acc)
    print(
        f"\nTesting accuracy = {acc:.2f}%; F1 = {100.*f1:.2f}%; \
            Precision = {100.*prec:.2f}%;\
            Recall = {100.*rec:.2f}% \n\tLoss: {testloss:1.2e}\n")
    test_losses.append(loss.data.item())
    test_accs.append(acc)
    # t.close()


def adjust_learning_rate(optimizer, epoch):
    update_list = [80, 160]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * 0.1, 1e-8)
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
for epoch in tqdm(range(100)):
    lr = get_lr()
    # adjust_learning_rate(optimizer, epoch)
    print(f" Epoch: {epoch}, learning rate = {lr:1.2e};\n")

    train(epoch, model_harmonic)

    test(epoch, model_harmonic)
    gc.collect()

    # t.update(epoch)
# t.close()

plt.figure(figsize=(12, 8))
plt.plot(test_losses, label="Test Loss")
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Train/Test losses.")
plt.legend()
plt.savefig('train_test_losses')
plt.figure(figsize=(12, 8))
plt.plot(test_accs, label="Test")
plt.plot(train_accs, label="Train")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Train/Test accuracies.")
plt.legend()
plt.savefig('train_test_accuracies')
