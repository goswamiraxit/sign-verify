import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from tqdm import tqdm

from dataset import SignDataset
from models import ConvNet, ResConvNet, VggConvNet

import config

train_dataset = SignDataset('./data/sign_data/converted-train-data.csv', './data/sign_data/train')
train_dataloader = data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE)
test_dataset = SignDataset('./data/sign_data/test_data.csv', './data/sign_data/test')
test_dataloader = data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE)


def train_epoch(epoch, model, criterion, optimizer, dataloader):
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_batch_size = 0.0

    with tqdm(dataloader, unit='batch') as pbar:
        pbar.set_description('Training Epoch: {}'.format(epoch))
        for inps, tgts in pbar:
            # print('Input: {}, Target: {}'.format(inps.size(), tgts.size()))
            preds = model(inps)
            
            loss = criterion(preds, tgts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().item()
            
            pred_labels = preds.max(dim=-1)[1]
            acc = (pred_labels == tgts).sum().item()
            total_acc += acc
            total_batch_size += tgts.size(0)
            pbar.set_postfix(loss = total_loss, accuracy=100. * total_acc / total_batch_size)
    
    return total_loss, total_acc / total_batch_size


def valid_epoch(epoch, model, criterion, dataloader):
    model.eval()
    with T.no_grad():
        total_loss = 0.0
        total_acc = 0.0
        total_batch_size = 0.0

        with tqdm(dataloader, unit='batch') as pbar:
            pbar.set_description('Validation Epoch: {}'.format(epoch))
            for inps, tgts in pbar:
                # print('Input: {}, Target: {}'.format(inps.size(), tgts.size()))
                preds = model(inps)
                
                loss = criterion(preds, tgts)

                total_loss += loss.cpu().item()
                
                pred_labels = preds.max(dim=-1)[1]
                acc = (pred_labels == tgts).sum().item()
                total_acc += acc
                total_batch_size += tgts.size(0)
                pbar.set_postfix(loss = total_loss, accuracy=100. * total_acc / total_batch_size)
        
        return total_loss, total_acc / total_batch_size

# model = ConvNet()
model = ResConvNet()
# model = VggConvNet()

model.to(config.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)


for epoch in range(1, 41):
    train_loss, train_acc = train_epoch(epoch, model, criterion, optimizer, train_dataloader)
    valid_loss, valid_acc = valid_epoch(epoch, model, criterion, test_dataloader)

    print('Epoch: {}'.format(epoch))
    print('Train: Loss={:10.05f} Accuracy={:7.02f}%'.format(train_loss, 100. * train_acc))
    print('Valid: Loss={:10.05f} Accuracy={:7.02f}%'.format(valid_loss, 100. * valid_acc))
    print('')

    with open('metrics.txt', 'a') as fp:
        fp.write('Epoch: {}\n'.format(epoch))
        fp.write('Train: Loss={:10.05f} Accuracy={:7.02f}%\n'.format(train_loss, 100. * train_acc))
        fp.write('Valid: Loss={:10.05f} Accuracy={:7.02f}%\n'.format(valid_loss, 100. * valid_acc))
        fp.write('\n')
    
    exp_lr_scheduler.step()
    
    T.save(model.state_dict(), './checkpoints/model-epoch-{}.pt'.format(epoch))





