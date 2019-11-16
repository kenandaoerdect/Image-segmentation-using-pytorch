import os
import model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from load_img import MyDataset
from torchvision import transforms
from torch.utils.data import DataLoader


batchsize = 8
epochs = 50
train_data_path = 'data/train'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
bag = MyDataset(train_data_path, transform)
dataloader = DataLoader(bag, batch_size=batchsize, shuffle=True)


device = torch.device('cuda')
net = model.Net().to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.7)

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

for epoch in range(1, epochs+1):
    for batch_idx, (img, lab) in enumerate(dataloader):
        img, lab = img.to(device), lab.to(device)
        output = torch.sigmoid(net(img))
        loss = criterion(output, lab)

        output_np = output.cpu().data.numpy().copy()
        output_np = np.argmin(output_np, axis=1)
        y_np = lab.cpu().data.numpy().copy()
        y_np = np.argmin(y_np, axis=1)

        if batch_idx % 20 == 0:
            print('Epoch:[{}/{}]\tStep:[{}/{}]\tLoss:{:.6f}'.format(
                epoch, epochs, (batch_idx+1)*len(img), len(dataloader.dataset), loss.item()
            ))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        torch.save(net, 'checkpoints/model_epoch_{}.pth'.format(epoch))
        print('checkpoints/model_epoch_{}.pth saved!'.format(epoch))