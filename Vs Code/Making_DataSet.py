# CSV 추출 필요 라이브러리
import os
import pandas as pd
import time
from tqdm import tqdm
from tqdm import notebook
from PIL import Image
import numpy as np

# ======================= 폴더 생성 필요 라이브러리 및 초기값 ======================= #
import cv2
import os 

DataSet_path = "./Taekwondo/DataSet" 
train_path = "./Taekwondo/train"
test_path = "./Taekwondo/test" 
Actionfolder_name = "/A"          # 맨 앞에 / 존재 
train_image_folder = "/train_Image"
test_image_folder = "/test_Image"
Action_Num = [1,4,5,6,7,8,10,14]

os.mkdir(DataSet_path)                      # DataSet 폴더 생성
os.mkdir(DataSet_path + train_image_folder) # train_Image 폴더 생성
os.mkdir(DataSet_path + test_image_folder)  # test_Image 폴더 생성

# ======================= 프레임 추출 필요 라이브러리 ===================== #
from tqdm import notebook
import time

count=0
# =========================== 폴더 생성 ============================ #
for num in Action_Num:
    # print("Iterating(반복 중): " + "A" + str(num).zfill(2) + " folder")
    # for문으로 A01~A14까지 생성
    Dataset_train_Image_folder = DataSet_path + train_image_folder + Actionfolder_name + str(num).zfill(2)
    Dataset_test_Image_folder = DataSet_path + test_image_folder + Actionfolder_name + str(num).zfill(2)

    os.mkdir(Dataset_train_Image_folder)        # for문으로 train_Image 내부에 A01~A14 폴더 생성
    os.mkdir(Dataset_test_Image_folder)         # for문으로 test_Image 내부에 A01~A14 폴더 생성
    # 비디오 경로
    train_Video_path = train_path + Actionfolder_name + str(num).zfill(2)
    test_Video_path = test_path + Actionfolder_name + str(num).zfill(2)
    for file in notebook.tqdm(os.listdir("A"+ str(num).zfill(2)), desc = 'Extracting train_Image from  A' + str(num).zfill(2)):        # 로딩바 & 파일이름 변수file에 저장 
        video = cv2.VideoCapture(os.path.join(train_Video_path, file))              # 비디오 경로
        # 비디오는 해당 코드와 같은 폴더에 있어야 한다.
        while(video.isOpened()):
            ret, image = video.read()
            if(video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT)):  # 현재프레임과 총프레임 비교
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)                                       # 현재 프레임 0으로 초기화
                cv2.imwrite(f'./Taekwondo/DataSet/train_Image/{ "A" + str(num).zfill(2) }/frame{count}.png', image)    # 추출
                count += 1
                break
    for file in notebook.tqdm(os.listdir("A"+ str(num).zfill(2)), desc = 'Extracting test_Image from  A' + str(num).zfill(2)):        # 로딩바 & 파일이름 변수file에 저장 
        video = cv2.VideoCapture(os.path.join(test_Video_path, file))              # 비디오 경로
        # 비디오는 해당 코드와 같은 폴더에 있어야 한다.
        while(video.isOpened()):
            ret, image = video.read()
            if(video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT)):  # 현재프레임과 총프레임 비교
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)                                       # 현재 프레임 0으로 초기화
                cv2.imwrite(f'./Taekwondo/DataSet/test_Image/{ "A" + str(num).zfill(2) }/frame{count}.png', image)    # 추출
                count += 1
                break

# ========================== train 프레임 추출 =========================== #
# ========================== train 프레임 추출 =========================== #
#================================여기까진 성공============================
# ========================== CSV 추출 =========================== #
columnNames = list()
columnNames.append('label')

for i in range(784):
    pixel = str(i)
    columnNames.append(pixel)
train_data = pd.DataFrame(columns = columnNames)
num_images = 0

for num in Action_Num:
    # print("Iterating: " + str(i) + " folder")
    for file in notebook.tqdm(os.listdir(Dataset_train_Image_folder), desc = 'Extracting CSV from  A' + str(num).zfill(2)):
        img = Image.open(os.path.join(Dataset_train_Image_folder, file))
        img = img.resize((28, 28), Image.NEAREST)
        img.load()
        imgdata = np.asarray(img, dtype="int32")
       
        data = []
        data.append(str(i))
        for y in range(28):
            for x in range(28):
                data.append(imgdata[x][y])

        train_data.loc[num_images] = data

        num_images += 1

train_data.to_csv("train_converted.csv", index=False)

# ========================== CSV 추출 =========================== #

columnNames = list()
columnNames.append('label')

for i in range(784):
    pixel = str(i)
    columnNames.append(pixel)
test_data = pd.DataFrame(columns = columnNames)
num_images = 0

for num in Action_Num:
    # print("Iterating: " + str(i) + " folder")
    for file in notebook.tqdm(os.listdir(Dataset_test_Image_folder), desc = 'Extracting CSV from  A' + str(num).zfill(2)):
        img = Image.open(os.path.join(Dataset_test_Image_folder, file))
        img = img.resize((28, 28), Image.NEAREST)
        img.load()
        imgdata = np.asarray(img, dtype="int32")
       
        data = []
        data.append(str(i))
        for y in range(28):
            for x in range(28):
                data.append(imgdata[x][y])

        test_data.loc[num_images] = data

        num_images += 1

test_data.to_csv("train_converted.csv", index=False)