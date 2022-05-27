# CSV 추출 필요 라이브러리
import os
import pandas as pd
import time
from tqdm import tqdm
from tqdm import notebook
from PIL import Image
import numpy as np

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
    for file in notebook.tqdm(os.listdir("A"+ str(num).zfill(2)), desc = 'Extracting CSV from  A' + str(num).zfill(2)):
        img = Image.open(os.path.join(Dataset_train_csv_path, file))
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