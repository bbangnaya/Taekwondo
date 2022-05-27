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


# 2. 딥러닝 모델 설계할 때 활용하는 장비 확인
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


class CNN(nn.Module):                 # nn.Module을 상속받는 Net클래스 생성
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
        # 2차원 이미지 데이터를 nn.Conv2d메서드를 이용해 Convolution연산을 하는 filter를 정의
        
        # in_channels = 3
        # 채널 수를 이미지의 채널수와 맞춰야 한다. RGB 채널은 채널수 = 3 이다.
        
        # out_channels = 8
        # Convolution연산을 진행하는 필터 개수
        # 여기서 설정해주는 Filter개수만큼 Output의 depth가 정해집니다.
        # 여기서는 depth가 8인 Feature Map이 생성
        
        # kernel_size = 3
        # Filter의 크기. 스칼라 값으로 설정하려면 가로 * 세로 크기인 Filter를 이용.
        # kernel_size = 3이면 3*3 의 필터가 이미지 위를 돌아다니면서 겹치는 영역에 대해
        # 9개의 픽셀 값과  Filter내에 있는 9개의 파라미터 값을 Convolution연산으로 진행.
    
        self.pool = nn.MaxPool2d(kernel_size = 2,
                                 stride = 2)
        self.fc1=nn.Linear(1024,128) # (input node수, output node수)
        self.fc2=nn.Linear(128,8)   # 이전 output node수 = 다음 input node수
        
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
        # 최종 생성된 Feature Map 의 모양은 8*8*16 크기이므로 이를 펼치기 위해 view 함수를 이용해 
        # Feature Map을 1차원 데이터로 변환
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x

model=CNN().to(DEVICE)                  # MLP모델을 기존에 선정한 'DEVICE'에 할당합니다. 
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
# Back Propagation을 이용해 파라미터를 업데이트할 때 이용하는 Optimizer를 정의합니다.
# SGD알고리즘을 이용하며 Learning Rate = 0.01, momentum=0.5로 설정
criterion=nn.CrossEntropyLoss()
# MLP모델의 output값과 계산될 Label값은 Class를 표현하는 원-핫 인코딩 값입니다.
# MLP모델의 output값과 원-핫 인코딩 값과의 Loss는 CrossEntropy를 이용해 계산하기 위해
# criterion은 nn.CrossEntropyLoss() 로 설정. 

print(model)


def train(model, train_loader, optimizer, log_interval):
    model.train()         # 학습상태로 지정
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
    model.eval()                  # 평가상태로 지정
    test_loss=0
    correct=0
    
    with torch.no_grad():
        for image, label in test_loader:
            image=image.to(DEVICE)              # 8과 동일
            label=label.to(DEVICE)              # 8과 동일
            output=model(image)                 # 8과 동일
            test_loss+=criterion(output,label).item()
            prediction = output.max(1,keepdim = True)[1]
            # MLP 모델의 output값은 크기가 10인 벡터값입니다. 
            # 계산된 벡터값 내 가장 큰 값인 위치에 대해 해당 위치에 대응하는 클래스로 예측했다고 판단합니다.
            correct += prediction.eq(label.view_as(prediction)).sum().item()
            # MLP모델이 최종으로 예측한 클래스 값과 실제 레이블이 의미하는 클래스가 맞으면 correct에 더해 올바르게 예측한 횟수를 저장
                        
    test_loss /= len(test_loader.dataset)
    # 현재까지 계산된 test_loss 값을 test_loader내에 존재하는 Mini-Batch 개수(=10)만큼 나눠 평균 Loss값으로 계산.
    test_accuracy=100.*correct / len(test_loader.dataset)
    # test_loader 데이터중 얼마나 맞췄는지를 계산해 정확도를 계산합니다.
    return test_loss, test_accuracy


for Epoch in range(1,EPOCHS + 1):
    train(model,train_loader,optimizer,log_interval = 200)
    test_loss, test_accuracy=evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".
        format(Epoch,test_loss,test_accuracy))








