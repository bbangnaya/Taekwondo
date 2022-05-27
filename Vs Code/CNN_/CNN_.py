
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
from torchvision import transforms, datasets
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
        image = read_image(img_path).float()
        label = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
     

train_data = CustomImageDataset("./train.csv",
                                   "../../Taekwondo/DataSet/train_Image",
                                   None, None)
test_data = CustomImageDataset("./test.csv",
                               "../../Taekwondo/DataSet/test_Image",
                               None, None)
train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                           batch_size = BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                        batch_size = BATCH_SIZE,
                                        shuffle=True)


# 6. 
for(X_train,Y_train)in train_loader:
    print('X_train:',X_train.size(),'type:', X_train.type())
    print('Y_train:',Y_train.size(),'type:', Y_train.type())
    break

#7. ������ Ȯ���ϱ�
pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)     # ���� �׷��� �׸���, ù���� : ��, ��° : ��
    plt.axis('off')           # �����
    plt.imshow(np.transpose(X_train[i],(1,2,0)))
    plt.title('Class: ' + str(Y_train[i].item()))
# imshow : �̹��� ���

# 8. CNN ����
class CNN(nn.Module):    
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
    
        self.pool = nn.MaxPool2d(kernel_size = 2,
                                 stride = 2)
        self.fc1=nn.Linear(1024,128) 
        self.fc2=nn.Linear(128,8) 
        
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
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x


model=CNN().to(DEVICE)                 
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion=nn.CrossEntropyLoss()

print(model)


def train(model, train_loader, optimizer, log_interval):
    model.train()         
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
    model.eval()                  
    test_loss=0
    correct=0
    
    with torch.no_grad():
        for image, label in test_loader:
            image=image.to(DEVICE)              
            label=label.to(DEVICE)              
            output=model(image)                 
            test_loss+=criterion(output,label).item()
            prediction = output.max(1,keepdim = True)[1]
            
            correct += prediction.eq(label.view_as(prediction)).sum().item()
            
                        
    test_loss /= len(test_loader.dataset)
    test_accuracy=100.*correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for Epoch in range(1,EPOCHS + 1):
    train(model,train_loader,optimizer,log_interval = 200)
    test_loss, test_accuracy=evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".
        format(Epoch,test_loss,test_accuracy))

