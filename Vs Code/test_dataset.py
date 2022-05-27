#1. module import
from ast import increment_lineno
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
import pandas as pd
from torchvision.io import read_image


# 2. ������ �� ������ �� Ȱ���ϴ� ��� Ȯ��
if torch.cuda.is_available():
     DEVICE=torch.device('cuda')
else:
     DEVICE=torch.device('cpu')

print('Using Pytorch version:',torch.__version__,'Device:',DEVICE)

BATCH_SIZE = 16
EPOCHS = 200

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        

training_data = CustomImageDataset("./train.csv", "./Taekwondo/DataSet/train_Image", None, None)
test_data = CustomImageDataset("./test.csv", "./Taekwondo/DataSet/test_Image", None, None)

train_loader = DataLoader(training_data, batch_size=128,

                          shuffle=True, num_workers=4)

test_loader = DataLoader(test_data, batch_size=128,

                          shuffle=False, num_workers=4)


class CNN(nn.Module):                 # nn.Module�� ��ӹ޴� NetŬ���� ����
    def __init__(self):   
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, 
                               out_channels = 16, 
                               kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 16, 
                               out_channels = 32, 
                               kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 32, 
                               out_channels = 64, 
                               kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 64, 
                               out_channels = 128, 
                               kernel_size = 3)
        # 2���� �̹��� �����͸� nn.Conv2d�޼��带 �̿��� Convolution������ �ϴ� filter�� ����
        
        # in_channels = 3
        # ä�� ���� �̹����� ä�μ��� ����� �Ѵ�. RGB ä���� ä�μ� = 3 �̴�.
        
        # out_channels = 8
        # Convolution������ �����ϴ� ���� ����
        # ���⼭ �������ִ� Filter������ŭ Output�� depth�� �������ϴ�.
        # ���⼭�� depth�� 8�� Feature Map�� ����
        
        # kernel_size = 3
        # Filter�� ũ��. ��Į�� ������ �����Ϸ��� ���� * ���� ũ���� Filter�� �̿�.
        # kernel_size = 3�̸� 3*3 �� ���Ͱ� �̹��� ���� ���ƴٴϸ鼭 ��ġ�� ������ ����
        # 9���� �ȼ� ����  Filter���� �ִ� 9���� �Ķ���� ���� Convolution�������� ����.
    
        self.pool = nn.MaxPool2d(kernel_size = 2,
                                 stride = 2)
        self.fc1=nn.Linear(1024,128) # (input node��, output node��)
        self.fc2=nn.Linear(128,8)   # ���� output node�� = ���� input node��
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.view(-1, 8 * 8 * 16)
        # ���� ������ Feature Map �� ����� 8*8*16 ũ���̹Ƿ� �̸� ��ġ�� ���� view �Լ��� �̿��� 
        # Feature Map�� 1���� �����ͷ� ��ȯ
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x

model=CNN().to(DEVICE)                  # MLP���� ������ ������ 'DEVICE'�� �Ҵ��մϴ�. 
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
# Back Propagation�� �̿��� �Ķ���͸� ������Ʈ�� �� �̿��ϴ� Optimizer�� �����մϴ�.
# SGD�˰����� �̿��ϸ� Learning Rate = 0.01, momentum=0.5�� ����
criterion=nn.CrossEntropyLoss()
# MLP���� output���� ���� Label���� Class�� ǥ���ϴ� ��-�� ���ڵ� ���Դϴ�.
# MLP���� output���� ��-�� ���ڵ� ������ Loss�� CrossEntropy�� �̿��� ����ϱ� ����
# criterion�� nn.CrossEntropyLoss() �� ����. 

print(model)


def train(model, train_loader, optimizer, log_interval):
    model.train()         # �н����·� ����
    for batch_idx,(image, label) in enumerate(train_loader):
        image=image.to(DEVICE)
        label=label.to(DEVICE)
        optimizer.zero_grad()
        output=model(image)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval==0:
            print("Train Epoch: {}[{}/{}({:.0f}%)]\tTrain Loss: {:.6f}"
                  .format(Epoch,batch_idx * len(image),
                len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))

def evaluate(model,test_loader):
    model.eval()                  # �򰡻��·� ����
    test_loss=0
    correct=0
    
    with torch.no_grad():
        for image, label in test_loader:
            image=image.to(DEVICE)              # 8�� ����
            label=label.to(DEVICE)              # 8�� ����
            output=model(image)                 # 8�� ����
            test_loss+=criterion(output,label).item()
            prediction = output.max(1,keepdim = True)[1]
            # MLP ���� output���� ũ�Ⱑ 10�� ���Ͱ��Դϴ�. 
            # ���� ���Ͱ� �� ���� ū ���� ��ġ�� ���� �ش� ��ġ�� �����ϴ� Ŭ������ �����ߴٰ� �Ǵ��մϴ�.
            correct += prediction.eq(label.view_as(prediction)).sum().item()
            # MLP���� �������� ������ Ŭ���� ���� ���� ���̺��� �ǹ��ϴ� Ŭ������ ������ correct�� ���� �ùٸ��� ������ Ƚ���� ����
                        
    test_loss /= len(test_loader.dataset)
    # ������� ���� test_loss ���� test_loader���� �����ϴ� Mini-Batch ����(=10)��ŭ ���� ��� Loss������ ���.
    test_accuracy=100.*correct / len(test_loader.dataset)
    # test_loader �������� �󸶳� ��������� ����� ��Ȯ���� ����մϴ�.
    return test_loss, test_accuracy


for Epoch in range(1,EPOCHS + 1):
    train(model,train_loader,optimizer,log_interval = 200)
    test_loss, test_accuracy=evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".
        format(Epoch,test_loss,test_accuracy))








