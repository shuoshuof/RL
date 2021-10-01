from vehicle import Driver
from controller import Camera,DistanceSensor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import torch
from torch import nn
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from PIL import Image

import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, models, datasets
np.set_printoptions(threshold=np.inf)
found_data = np.zeros((64, 13), dtype=int) # 【0左边界像素数，1右边界像素数，2左边界坐标，3右边界坐标，4左补线后的边界情况（0为缺失），5右补线后的边界情况（0为缺失），6赛道宽度，7层数,8扫描起始点】
imgresult=[]
l_left=0
r_left=0
last_left_l=0
last_left_r=0
last_right_l=0
last_right_r=0
l_right=0
r_right=0
r_left0=0
r_right0=0
Use_Line=0
Use_ROWS=0
left_found=0
right_found=0
left_end=0
right_end=0
last_middle_point=int(Use_Line/2)

def eight_adjacent_sides(i, j,Pixle):
    return Pixle[i][j + 1] + Pixle[i][j - 1] + Pixle[i - 1][j + 1] + Pixle[i - 1][j - 1] + \
           Pixle[i + 1][j + 1] + Pixle[i + 1][j - 1] + Pixle[i - 1][j] + Pixle[i + 1][j]
def eight_adjacent_sides_left_finding(Pixle):
    global r_left,l_left,imgresult,last_left_l,last_left_r,r_left0,found_data,left_found,left_end,crossroads,crossroads_left_condition,crossroads_right_condition
    left_found=0
    if  (l_left >3 and l_left<Use_Line-3) :# l 太大或太小，说明其已经接近边界，所以这边可以不检测了，除非是比较靠近车头出现此种情况
        if Pixle[r_left - 1][l_left + 1] == 1 :  # 2
            r_left, l_left = r_left, l_left + 1
            imgresult[r_left][l_left] = 200
        if Pixle[r_left - 1][l_left] == 1 and Pixle[r_left - 1][l_left + 1] == 0:  # 3
            r_left, l_left = r_left - 1, l_left + 1
            imgresult[r_left][l_left] = 200
        elif Pixle[r_left - 1][l_left - 1] == 1 and Pixle[r_left - 1][l_left] == 0:  # 4
            r_left, l_left = r_left - 1, l_left
            imgresult[r_left][l_left] = 200
        elif Pixle[r_left][l_left - 1] == 1 and Pixle[r_left - 1][l_left - 1] == 0:  # 5
            r_left, l_left = r_left - 1, l_left - 1
            imgresult[r_left][l_left] = 200
        elif Pixle[r_left + 1][l_left - 1] == 1 and Pixle[r_left][l_left - 1] == 0:  # 6
            r_left, l_left = r_left, l_left - 1
            imgresult[r_left][l_left] = 200
        elif Pixle[r_left + 1][l_left] == 1 and Pixle[r_left+1][l_left - 1] == 0:
            r_left, l_left =r_left + 1,l_left - 1
            imgresult[r_left][l_left] = 200
        if last_left_r == r_left and last_left_l == l_left:#说明卡住
            r_left0 = r_left0-1#往上一行扫描
            found_data[r_left0][9] = 1
            for j in range(found_data[r_left0+1][3] - 20, 20, -1):#从右边点附近开始扫描
                imgresult[r_left0][j] = 150
                if eight_adjacent_sides(r_left0, j, Pixle) in range(2, 5):#找到边界才允许r_left,l_left更新
                    l_left = j
                    r_left = r_left0
                    imgresult[r_left0][j] = 150
                    break
            left_found=0
        else:
            if r_left<r_left0:
                r_left0=r_left
                found_data[r_left0][4]=l_left
            if r_left==r_left0: #防止弯道使r_left变大
                found_data[r_left0][0] += 1
                found_data[r_left0][2] = l_left#记录的是该行最靠边的点坐标
            last_left_r = r_left
            last_left_l = l_left
            left_found=1
    elif r_left0>25: #对于接近车头延伸到边界的边线，认为有继续检测的价值，这种情况很可能是环岛的入口缺边，继续检测再次发现边线，而不是结束
        r_left0 = r_left0 - 1
        left_end=1
        found_data[r_left0][11] = 1
        for j in range(found_data[r_left0 + 1][3] - 10, 20, -1):
            imgresult[r_left0][j] = 150
            if eight_adjacent_sides(r_left0, j, Pixle) in range(2, 5):
                l_left = j
                r_left = r_left0
                break
    else:#太靠边的点标志结束
        left_end=1
        r_left0-=1
def eight_adjacent_sides_right_finding(Pixle):
    global r_right,l_right,last_right_l,last_right_r,last_right_l,r_right0,right_add,found_data,right_found,right_end,crossroads,crossroads_right_condition,crossroads_left_condition
    right_found=0
    if (l_right>0 and l_right <Use_Line-3) :
        if Pixle[r_right - 1][l_right - 1] == 1 :  # 2
            r_right, l_right = r_right, l_right - 1
            imgresult[r_right][l_right] = 255
        elif Pixle[r_right - 1][l_right] == 1 and Pixle[r_right - 1][l_right - 1]==0:  # 3
            r_right, l_right = r_right - 1, l_right - 1
            imgresult[r_right][l_right] = 255
        elif Pixle[r_right - 1][l_right + 1] == 1 and Pixle[r_right - 1][l_right]==0:  # 4
            r_right, l_right = r_right - 1, l_right
            imgresult[r_right][l_right] = 255
        elif Pixle[r_right][l_right + 1] == 1 and Pixle[r_right - 1][l_right + 1] ==0:  # 5
            r_right, l_right = r_right - 1, l_right + 1
            imgresult[r_right][l_right] = 255
        elif Pixle[r_right + 1][l_right + 1] == 1 and Pixle[r_right][l_right + 1] == 0:  # 6
            r_right, l_right = r_right, l_right + 1
            imgresult[r_right][l_right] = 255
        elif Pixle[r_right+1][l_right]==1 and Pixle[r_right + 1][l_right + 1]==0:
            r_right, l_right= r_right+1, l_right + 1
            imgresult[r_right][l_right] = 255
        if last_right_r == r_right and last_right_l == l_right:
            r_right0=r_right0-1
            found_data[r_right0][10] = 1
            for j in range(found_data[r_right0+1][2]+30,Use_Line-20):
                imgresult[r_right0][j] = 100

                if eight_adjacent_sides(r_right0, j, Pixle) in range(2, 5):
                    l_right = j
                    r_right=r_right0
                    break
            right_found=0

        else:
            if r_right<r_right0:
                found_data[r_right][5]=l_right
                r_right0=r_right
            if r_right==r_right0:
                found_data[r_right0][1] += 1
                found_data[r_right0][3] =l_right
            last_right_r = r_right
            last_right_l = l_right
            right_found=1
    elif r_right>25:
        right_end=1
        r_right0 = r_right0 - 1
        found_data[r_right0][12] = 1
        for j in range(found_data[r_right0 + 1][2] + 10, Use_Line - 20):
            imgresult[r_right0][j] = 100
            if eight_adjacent_sides(r_right0, j, Pixle) in range(2, 5):
                l_right = j
                r_right = r_right0
                break
    else:
        right_end=1
        r_right0-=1
def eight_adjacent_sides_find(Pixle):
    global found_data,r_right,r_left,l_left,l_right,imgresult,r_right0,r_left0,crossroads,crossroads_left_condition,crossroads_right_condition
    global Use_ROWS,Use_Line,crossroads_middle_line,crossroad_flag
    Use_ROWS,Use_Line=Pixle.shape
    found_data = np.zeros((64, 13), dtype=np.uint8)
    for i in range(Use_ROWS):
        found_data[i][7]=i
    imgresult = np.zeros(Pixle.shape,dtype=np.uint8)
    r_left,l_left=0,0
    r_left0=Use_ROWS-20
    r_right,l_right=0,Use_Line-1
    r_right0=Use_ROWS-20
    #初始行赋值

    while l_left==0:
        r_left0-=1
        for j in range(Use_Line//2,1,-1):
            if eight_adjacent_sides(r_left0,j,Pixle) in range(2,5):
                r_left,l_left=r_left0,j
                found_data[r_left0][2]=l_left
                print("left_found")
                print(l_left)
                break
    while l_right==Use_Line-1:
        r_right0-=1
        for j in range(Use_Line//2,Use_Line-1):
            if eight_adjacent_sides(r_right0,j,Pixle) in range(2,5):
                r_right,l_right=r_right0,j
                found_data[r_right0][3]=l_right
                break
    found_data[r_left0][8] = (found_data[r_left0][2] + found_data[r_right0][3]) / 2
    found_data[r_left0][6] = found_data[r_right0][3] - found_data[r_left0][2]

    last_right_rows=0
    last_left_rows=0
    right_time=0
    left_time=0
    error_flag=0
    while r_right0>20 and r_left0>20 and (left_end==0 or right_end==0) and (l_left<l_right):
        if last_right_rows!=r_right0:
            last_right_rows=r_right0
            right_time=0
        else:
            right_time+=1
        if last_left_rows!=r_left0:
            last_left_rows=r_left0
            left_time=0
        else:
            left_time+=1
        if left_time>80 or right_time>80:
            r_left0-=1
            left_time=0
            r_right0-=1
            right_time=0
        if left_end==0 and right_end==0:
            if r_right0==r_left0:
                eight_adjacent_sides_left_finding(Pixle)
                eight_adjacent_sides_right_finding(Pixle)
            elif r_left0<r_right0:
                eight_adjacent_sides_right_finding(Pixle)
            elif r_right0<r_left0:
                eight_adjacent_sides_left_finding(Pixle)
        else:
            if right_end:
                eight_adjacent_sides_left_finding(Pixle)
            else:
                eight_adjacent_sides_right_finding(Pixle)
        if r_left0 == r_right0:
            if left_found and right_found:
                found_data[r_left0][8]=(found_data[r_left0][2]+found_data[r_left0][3])/2
                found_data[r_left0][6]=found_data[r_left0][3]-found_data[r_left0][2]
            elif left_found==0 and right_found:
                found_data[r_left0][8]=found_data[r_left0][3]-found_data[r_left0+1][3]+found_data[r_left0+1][8]
            elif left_found and right_found==0:
                found_data[r_left0][8]=found_data[r_left0][2]-found_data[r_left0+1][2]+found_data[r_left0+1][8]
    #     plt.cla()
    #     plt.imshow(imgresult)
    #     plt.pause(0.01)
    #     print("r,l=",r_right0,r_left0,right_time,left_time)
    # print(found_data)
    return imgresult


net = nn.Sequential(
    nn.Conv2d(3, 24, kernel_size=5, padding=0, stride=(2, 2)),
    nn.ELU(),
    # nn.Dropout(p=0.2),
    nn.Conv2d(24, 36, kernel_size=5, padding=0, stride=(2, 2)),
    nn.ELU(),
    # nn.Dropout(p=0.2),
    nn.Conv2d(36, 48, kernel_size=5, padding=0, stride=(2, 2)),
    nn.ELU(),
    # nn.Dropout(p=0.2),
    nn.Conv2d(48, 64, kernel_size=3, padding=0),
    nn.ELU(),
    # nn.Dropout(p=0.2),
    nn.Conv2d(64, 64, kernel_size=3, padding=0),
    nn.ELU(),
    # nn.Dropout(p=0.2),
    nn.Flatten(),
    nn.Linear(1280, 100),
    nn.ELU(),
    # nn.Dropout(p=0.5),
    nn.Linear(100, 50),
    nn.ELU(),
    # nn.Dropout(p=0.5),
    nn.Linear(50, 10),
    nn.ELU(),
    # nn.Dropout(p=0.5),
    nn.Linear(10, 1)

)
sensorMax = 1000
driver = Driver()
ds = []
dsNames = [
    'ds1', 'ds2','ds3','ds4','ds5','ds6','ds7','ds8'
]

basicTimeStep = int(driver.getBasicTimeStep())
camera = Camera('camera')
camera.enable(100)
sensorTimeStep = 4 * basicTimeStep
for i in range(len(dsNames)):
    ds.append(driver.getDevice(dsNames[i]))
    ds[i].enable(basicTimeStep)
speed = 0
angle = 0
maxSpeed = 1.8
driver.setSteeringAngle(angle)
driver.setCruisingSpeed(speed)
printCounter = 0
last_error = 0
p_last_error = 0
def pid_calculate(error):
    p = 0.15
    i = 0
    d = 0.05
    global last_error
    pid_value = p*error + d*(error -last_error)
    last_error = error
    return pid_value
def error_calculate(dsValues):
    error = 0
    num =0
    for i in range(len(dsValues)):
        if dsValues[i]==1:
            error += i+1-3.5
            num +=1
    if num == 0:
        error=last_error
    else:
        error = error/num

    return error
def line_remove(img,gray):
    if gray:
        r,w =img.shape
        print(r,w)
        for i in range(r):
            for j in range(w):
                if img[i][j] <=25:
                    img[i][j]  =90

    else:
        r,w,h =img.shape
        for i in range(r):
            for j in range(w):
                if img[i][j][0] <=50 and img[i][j][1] <=50 and img[i][j][2] <=50:
                    img[i][j][0]  =230+random.randint(-5,5)
                    img[i][j][1] = 230+random.randint(-5,5)
                    img[i][j][2] = 230+random.randint(-5,5)
                for k in range(h):
                    if img[i][j][k] >255 :
                        img[i][j][k]  =255
                    if img[i][j][k] <0:
                        img[i][j][k] = 0
    return img
def img_pro(img,To_gray=False):
    img = np.random.randint(-10,10,img.shape)+img #添加噪声
    if To_gray:
        img = line_remove(img, True)
    else:
        img = line_remove(img, False)

    return img
def pixel_error_calculate(line):
    num = 0
    error =0
    for i in range(len(line)):
        if line[i][0] <=40 and line[i][1] <=40 and line[i][2] <=40:
            error+=i
            num+=1
    if num :
        error = error/num-(len(line)//2)
    else:
        error = p_last_error
    return error
def p_pid_calculate(error):
    p = 0.02
    i = 0
    d = 0.01
    global p_last_error
    pid_value = p*error + d*(error -p_last_error)
    p_last_error = error
    return pid_value
pre = False
save = True

if pre:
    model = net
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    # model = load_model('./model.h5')
    num = 0
elif save:
    num = int(input("input:"))
else :
    num=0
data_transform = {
    "train": transforms.Compose([
                                transforms.ToTensor()
                                ]),
    "val": transforms.Compose([
                               transforms.ToTensor()
                                ])}
transform=data_transform["train"]
while driver.step() != -1:
    # l_left = 0
    # r_left = 0
    # last_left_l = 0
    # last_left_r = 0
    # last_right_l = 0
    # last_right_r = 0
    # l_right = 0
    # r_right = 0
    # r_left0 = 0
    # r_right0 = 0
    # Use_Line = 0
    # Use_ROWS = 0
    # left_found = 0
    # right_found = 0
    # left_end = 0
    # right_end = 0
    printCounter += 1
    cameraData  = camera.getImage()
    # img = camera.getImageArray()
    img=np.frombuffer(cameraData, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))

    img = cv2.flip(img, 0)
    img = cv2.flip(img, 1)
    img = img[54:120,:,:3]

    # print(img.shape)
    plt.cla()
    plt.imshow(img)
    plt.pause(0.01)
    # print(img[100][10:128-10])
    # print(img[100][10:128-10].shape)
    line =img[60][20:200-20]

    img =img_pro(img)
    # print(img.shape)
    pixel_error = pixel_error_calculate(line)
    print("real_pixel_error=", pixel_error)
    p_angle = p_pid_calculate(pixel_error)
    # print("real_p_angle=",p_angle)
    dsValues = []
    r,w,h =img.shape
    for i in range(len(dsNames)):
        if ds[i].getValue() >120:
            dsValues.append(1)
        else:
            dsValues.append(0)
    if  pre==False :
        # print(dsValues)
        # error = error_calculate(dsValues)
        # print(error)
        # angle = pid_calculate(error)
        driver.setCruisingSpeed(3)
        driver.setSteeringAngle(p_angle)
        # print(angle)
        if num % 50 == 0:
            plt.cla()
            plt.imshow(img)
            plt.pause(0.01)
        if save :
            if num%10 == 0:
                # if angle<=0.05 and angle >=-0.05:
                #     if random.randint(0,10)>=7:
                #         cv2.imwrite("./test_dataset/{}_{:.2f}_{:.2f}.jpg".format(num,p_angle,pixel_error),img)
                # else:
                #     cv2.imwrite("./test_dataset/{}_{:.2f}_{:.2f}.jpg".format(num,p_angle,pixel_error),img)
                cv2.imwrite("./dataset3/train/{}_{:.2f}_{:.2f}.jpg".format(num, p_angle, pixel_error), img)
    elif pre:
        if num <= 30:
            driver.setCruisingSpeed(0)
            driver.setSteeringAngle(0)
        else:
            # img = np.array(img, dtype=np.uint8)
            # img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
            cv2.imwrite("./img.jpg", img)
            img = Image.open("./img.jpg")
            img = transform(img)
            img = img.unsqueeze(0)
            with torch.no_grad():

                # img = img.reshape(1,h,r,w)
                # img = torch.tensor(img, dtype=torch.float32)/255
                prediction = model(img)
                print(prediction)
                pixel_error = prediction.numpy()[0][0].astype(dtype=np.double)
                print("pixel_error=",pixel_error)
                p_angle = p_pid_calculate(pixel_error)
                # print("p_angle=",p_angle)
                driver.setCruisingSpeed(3)
                driver.setSteeringAngle(p_angle)
            # print(img.shape)

            # img = cv2.resize(img,(66,200))
            # img = np.array([img])
            # pixel_error = float(model.predict(img/255))
            # print("pixel_error=",pixel_error)
            # p_angle = p_pid_calculate(pixel_error)
            # driver.setCruisingSpeed(3)
            # driver.setSteeringAngle(p_angle)
    num +=1
