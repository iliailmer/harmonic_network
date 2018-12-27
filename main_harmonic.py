from model_harmonic import HarmonicNet
from loader import Loader
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torchvision
import os
from glob import glob

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
imgs_28_rgb = pd.read_csv('hmnist_28_28_RGB.csv')
labels = imgs_28_rgb['label'].values
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('HAM10000_images', '*.jpg'))}
del imgs_28_rgb


training_data_ = Loader(path='HAM10000_images',
                        labels=labels, train=True,
                        transform=torchvision.transforms.ToTensor())

testing_data_ = Loader(path='HAM10000_images',
                       labels=labels, train=False,
                       transform=torchvision.transforms.ToTensor())

model_harmonic = HarmonicNet(3, 32, 4, 2, 0)
for module in model_harmonic.modules():
    if isinstance(module, nn.Conv2d):
        module.weight.data.normal_(0, 0.05)
        module.bias.data.zero_()

model_harmonic.cuda()
model_harmonic = torch.nn.DataParallel(
    model_harmonic, device_ids=range(torch.cuda.device_count()))
base_lr = float(0.01)
param_dict = dict(model_harmonic.named_parameters())
params = []

loss = nn.CrossEntropyLoss()
# requires a small learning rate in the beginning, otherwise the loss blows up
optimizer = optim.SGD(model_harmonic.parameters(), lr=1e-5,
                      weight_decay=0.0005, momentum=0.9)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0


def train(epoch, model, output_step=100):
    model.train()
    losses = []
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = torch.autograd.Variable(
            data.cuda()), torch.autograd.Variable(label.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss_ = loss(output, label)

        loss_.backward()
        optimizer.step()
        losses.append(loss_.data.item())
        if batch_idx % output_step == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tTraining Loss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss_.data.item(),
                optimizer.param_groups[0]['lr']))
    return losses


def test(epoch, model, output_step=50):
    model.eval()
    losses = []
    for batch_id, (data, label) in enumerate(testloader):
        data, label = torch.autograd.Variable(
            data.cuda()), torch.autograd.Variable(label.cuda())
        output = model(data)
        test_loss_ = loss(output, label)
        losses.append(test_loss_.data.item())
        if batch_id % output_step == 0:
            print('Testing: {} [{}/{} ({:.0f}%)]\tTesting Loss: {:.6f}\tLR: {}'.format(
                epoch, batch_id * len(data), len(testloader.dataset),
                100. * batch_id / len(testloader), test_loss_.data.item(),
                optimizer.param_groups[0]['lr']))
    return losses


trainloader = torch.utils.data.DataLoader(training_data_, batch_size=64,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testing_data_, batch_size=64,
                                         shuffle=True, num_workers=2)
classes = list(lesion_type_dict.keys())

for epoch in range(15):
    train_losses = train(epoch, model_harmonic, 250)
    test_losses = test(epoch, model_harmonic, 250)
