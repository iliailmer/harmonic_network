from model_harmonic import HarmonicNet
from loader import Loader
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torchvision
import os
from glob import glob
from sklearn.preprocessing import LabelEncoder


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
metadata = metadata.sort_values(['image_id'],
                                ascending=True).reset_index(drop=True)
labels = metadata.dx.values

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('HAM10000_images', '*.jpg'))}


training_data_ = Loader(path='HAM10000_images', color_space='rgb',
                        labels=labels, train=True,
                        transform=torchvision.transforms.ToTensor())
testing_data_ = Loader(path='HAM10000_images', color_space='rgb',
                       labels=labels, train=False,
                       transform=torchvision.transforms.ToTensor())

model_harmonic = HarmonicNet(3, 18, 3, 2, 1)
# order of arguments for network, relate for the firts layer.
# input_channels,
# output_channels (from first layer... needs fixing?)
# kernel_size : size of the kernel of DCT
# stride
# padding

for module in model_harmonic.modules():
    if isinstance(module, nn.Conv2d):
        module.weight.data.normal_(0, 0.05)
        module.bias.data.zero_()

# for parameter in model_harmonic.parameters():
#  parameter.to('cuda')
model_harmonic.cuda()
model_harmonic = torch.nn.DataParallel(
    model_harmonic, device_ids=range(torch.cuda.device_count()))
base_lr = float(1e-2)
param_dict = dict(model_harmonic.named_parameters())
params = []

loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model_harmonic.parameters(), lr=base_lr)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0


trainloader = torch.utils.data.DataLoader(training_data_,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=2)
testloader = torch.utils.data.DataLoader(testing_data_,
                                         batch_size=64,
                                         shuffle=True,
                                         num_workers=2)
classes = list(lesion_type_dict.keys())


def train(epoch, model, output_step=128):
    model.train()
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = torch.autograd.Variable(
            data.cuda()), torch.autograd.Variable(label.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss_ = loss(output, label)
        _, y_pred = torch.max(output, 1)

        loss_.backward()
        optimizer.step()
        if batch_idx % output_step == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tTraining Loss: \
                    {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss_.data.item(),
                optimizer.param_groups[0]['lr']))
            train_losses.append(loss_.data.item())


def test(epoch, model):
    global best_acc
    model.eval()
    loss_ = 0
    test_loss_ = 0
    correct = 0
    accuracy = 0
    for batch_id, (data, label) in enumerate(testloader):
        data, label = torch.autograd.Variable(
            data.cuda()), torch.autograd.Variable(label.cuda())
        output = model(data)
        test_loss_ = loss(output, label)
        loss_ += loss(output, label).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        accuracy = 100. * correct / len(testloader.dataset)
    if accuracy > best_acc:
        best_acc = accuracy
        save_state(model, best_acc)
    print('Testing: {} [{}/{} ({:.0f}%)]\tTesting Loss: {:.3e}\tLR: {}'.format(
        epoch, batch_id * len(data), len(testloader.dataset),
        100. * batch_id / len(testloader), test_loss_.data.item(),
        optimizer.param_groups[0]['lr']))
    print(
        f'\nTest set: Average loss: \
        {loss_ * 128./len(testloader.dataset): .3f}\n')
    print(f'Accuracy: {correct}/{len(testloader.dataset):.3f}\
          ({100. * correct / len(testloader.dataset): .3f} %)')
    print(f'Best Accuracy: {best_acc:.2f}%\n')
    test_losses.append(test_loss_.data.item())


def adjust_learning_rate(optimizer, epoch):
    update_list = [20, 40, 80, 100, 160]  # [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
    return


def save_state(model, best_acc):
    print('\n==> Saving model ...')
    state = {
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
    }
    keys = list(state['state_dict'].keys())
    for key in keys:
        if 'module' in key:
            state['state_dict'][key.replace(
                'module.', '')] = state['state_dict'].pop(key)
    torch.save(state, 'harmonic_network.tar')


train_losses = []
test_losses = []

for epoch in range(200):
    adjust_learning_rate(optimizer, epoch)
    train(epoch, model_harmonic, 64)
    test(epoch, model_harmonic)
