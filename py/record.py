import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import os
import copy
# First method, the most general one
# First method, the most general one

class MyRes34(torch.nn.Module):
  # put your models here
  def __init__(self):
    super().__init__()
    self.res34 = torch.nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2])
    self.flatten = torch.nn.Flatten()
    self.fc1 = torch.nn.Linear(128*64, 512)
    self.relu1 = torch.nn.ReLU()
    self.dropout1 = torch.nn.Dropout(p=0.25)
    self.fc2 = torch.nn.Linear(512, 512)
    self.relu2 = torch.nn.ReLU()
    self.dropout2 = torch.nn.Dropout(p=0.25)
    self.fc3 = torch.nn.Linear(512, 4)
    self.softmax = torch.nn.Softmax(dim=1)
    
  # define inference
  def forward(self,x):
    x = self.res34(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    x = self.fc3(x)
    x = self.softmax(x)
#     x = self.relu(x)

    return x
    
class MyVGG(torch.nn.Module):
  # put your models here
  def __init__(self):
    super().__init__()
    self.vgg19_features = models.vgg19(pretrained=False).features
    self.flatten = torch.nn.Flatten()
    self.fc1 = torch.nn.Linear(128*64, 512)
    self.relu = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=0.25)
    self.fc2 = torch.nn.Linear(512, 512)
    self.relu2 = torch.nn.ReLU()
    self.dropout2 = torch.nn.Dropout(p=0.25)
    self.fc3 = torch.nn.Linear(512, 4)
    self.softmax = torch.nn.Softmax(dim=1)
    
  # define inference
  def forward(self,x):
    x = self.vgg19_features(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    x = self.fc3(x)
    x = self.softmax(x)

    return x


face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('xml/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('xml/haarcascade_mcs_nose.xml')


loaded_model = torch.load('beres4_model.h5', map_location=torch.device('cpu'))

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture('泡.gif')
cap3 = cv2.VideoCapture('ba.gif')
cap4 = cv2.VideoCapture('hartt.gif')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = cap.get(cv2.CAP_PROP_FPS)
fpss = fps*10
out = cv2.VideoWriter('output.mp4',fourcc, fpss, (1280,720))


while(cap.isOpened()):
    ret, frame = cap.read()
    frame=cv2.resize(frame, dsize=(1280, 720))
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not ret2:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    if not ret3:
        cap3.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    if not ret4:
        cap4.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame2=frame2[130:250,10:110,:]
        masks = np.where(((frame2!=[195,195,195]) & (frame2!=[194,194,194])).all(axis=2))
        maskb = np.where((frame3!=[0,0,0]).all(axis=2))
        maskh = np.where((frame4!=[255,255,300]).all(axis=2))
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            facecx=x+(w/2)
            
            height, width, channels =frame.shape[:3]
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray) 
            noses=nose_cascade.detectMultiScale(roi_gray) 
            
            img = cv2.resize(roi_color, dsize=(128, 128))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            trans = transforms.ToTensor()#np.rollaxis(img,2,0)
            img_2 = trans(img)
            img_2 = img_2[np.newaxis, :, :]
            # Predict
            output = loaded_model(img_2)
            _, pred = torch.max(output, 1)
            if pred==0:
                # define parameter
                HSV_MIN = np.array([0, 30, 60])
                HSV_MAX = np.array([20, 150, 255])
                if x-100<0:
                    lop=0
                else:
                    lop=x-100
                te1=frame[y:y+h, lop:x]
                te1_hsv=cv2.cvtColor(te1,cv2.COLOR_BGR2HSV)
                mask1_hsv = cv2.inRange(te1_hsv, HSV_MIN, HSV_MAX)
                cv2.imwrite("a.png",mask1_hsv)
                te2=frame[y:y+h, x+w:x+w+100]
                te2_hsv=cv2.cvtColor(te2,cv2.COLOR_BGR2HSV)
                mask2_hsv = cv2.inRange(te2_hsv, HSV_MIN, HSV_MAX)
                
                if (np.sum(mask1_hsv)+np.sum(mask2_hsv))/(h*200) > 40:
                    print("yubi")
                    counte=0
                    for (ex,ey,ew,eh) in eyes:
                        counte=counte+1
                    if counte ==2:
                        for (ex,ey,ew,eh) in eyes:
                            exc= int(ex + (ew/2))+x
                            eyc= int(ey + (eh/2))+y
                            if exc <= facecx:
                                cv2.line(frame, (exc, eyc), (0,eyc-100), (152, 145, 234), thickness=10, lineType=cv2.LINE_AA)
                                cv2.line(frame, (exc, eyc), (0,eyc-110), (152, 145, 234), thickness=10, lineType=cv2.LINE_AA)
                                cv2.line(frame, (exc, eyc), (0,eyc-90), (152, 145, 234), thickness=10, lineType=cv2.LINE_AA)
                                p=[]
                                q=[]
                                for j in range(5):
                                    p.append(exc*(1+j)//6)
                                    q.append((100)*(1+j)//6)
                                p.append(exc//3+20)
                                q.append((100)//3+10)
                                p.append(exc//3-20)
                                q.append((100)//3-10)
                                for i in range(7):
                                    if max(maskh[0],default=0)+q[i]+eyc-220> height or max(maskh[1],default=0)+p[i]-150> width:
                                        continue
                                    frame[maskh[0]+q[i]+eyc-220,maskh[1]+p[i]-150]=frame4[maskh[0],maskh[1]]
                            if exc > facecx:
                                cv2.line(frame, (exc, eyc), (width, eyc-100), (152, 145, 234), thickness=10, lineType=cv2.LINE_AA)
                                cv2.line(frame, (exc, eyc), (width, eyc-110), (152, 145, 234), thickness=10, lineType=cv2.LINE_AA)
                                cv2.line(frame, (exc, eyc), (width, eyc-90), (152, 145, 234), thickness=10, lineType=cv2.LINE_AA)
                                r=[]
                                l=[]
                                for j in range(5):
                                    r.append((exc-width)*(1+j)//6)
                                    l.append((100)*(1+j)//6)
                                r.append((exc-width)//3+20)
                                l.append((100)//3+10)
                                r.append((exc-width)//3-20)
                                l.append((100)//3-10)
                                for i in range(7):
                                    if max(maskh[0],default=0)+l[i]+eyc-220> height or max(maskh[1],default=0)+r[i]-100> width:
                                        continue
                                    frame[maskh[0]+l[i]+eyc-220,maskh[1]+r[i]-100]=frame4[maskh[0],maskh[1]]
                else:
                    print("no glasses")
                kind="no"
            if pred==2:
                print("skuea glasses")
                counte=0
                for (ex,ey,ew,eh) in eyes:
                    counte=counte+1
                    print("skuea glasses")
                if counte ==2:
                    for (ex,ey,ew,eh) in eyes:
                        exc= int(ex + (ew/2))+x
                        eyc= int(ey + (eh/2))+y
                        if exc <= facecx:
                            cv2.line(frame, (exc, eyc), (0,eyc-100), (254, 253, 200), thickness=5, lineType=cv2.LINE_AA)
                            p=[]
                            q=[]
                            for j in range(8):
                                p.append(exc*(1+j)//9)
                                q.append((100)*(1+j)//9)
                            p.append(exc//3+20)
                            q.append((100)//3+10)
                            p.append(exc//3-20)
                            q.append((100)//3-10)
                            for i in range(10):
                                if max(masks[0],default=0)+q[i]+eyc-170> height or max(masks[1],default=0)+p[i]-50> width:
                                    continue
                                frame[masks[0]+q[i]+eyc-170,masks[1]+p[i]-50]=frame2[masks[0],masks[1]]
                        if exc > facecx:
                            cv2.line(frame, (exc, eyc), (width, eyc-100), (254, 253, 200), thickness=5, lineType=cv2.LINE_AA)
                            r=[]
                            l=[]
                            for j in range(8):
                                r.append((exc-width)*(1+j)//9)
                                l.append((100)*(1+j)//9)
                            for i in range(8):
                                if max(masks[0],default=0)+l[i]+eyc-170> height or max(masks[1],default=0)+r[i]-50> width:
                                    continue
                                frame[masks[0]+l[i]+eyc-170,masks[1]+r[i]-50]=frame2[masks[0],masks[1]]
                        
            elif pred==1:
                print("round")
                counte=0
                for (ex,ey,ew,eh) in eyes:
                    counte=counte+1
                if counte ==2:
                    for (ex,ey,ew,eh) in eyes:
                        exc= int(ex + (ew/2))+x
                        eyc= int(ey + (eh/2))+y
                        if exc <= facecx:
                            cv2.circle(frame, (exc,eyc), 15,(65, 120, 220), thickness=30, lineType=cv2.LINE_AA, shift=0)
                            cv2.circle(frame, (exc,eyc), 13,(146, 253, 255), thickness=30, lineType=cv2.LINE_AA, shift=0)
                            cv2.circle(frame, (exc,eyc), 13,(204,234,234), thickness=15, lineType=cv2.LINE_AA, shift=0)
                            cv2.line(frame, (exc, eyc), (0,eyc-120), (146, 253, 255), thickness=35, lineType=cv2.LINE_AA)
                            cv2.line(frame, (exc, eyc), (0,eyc-80), (146, 253, 255), thickness=35, lineType=cv2.LINE_AA)
                            cv2.line(frame, (exc, eyc), (0,eyc-100), (146, 253, 255), thickness=35, lineType=cv2.LINE_AA)
                            p=[]
                            q=[]
                            for j in range(5):
                                p.append(exc*(1+j)//6)
                                q.append((100)*(1+j)//6)
                            p.append(exc//3+20)
                            q.append((100)//3+10)
                            p.append(exc//3-20)
                            q.append((100)//3-10)
                            for i in range(7):
                                if max(maskb[0],default=0)+q[i]+eyc-220> height or max(maskb[1],default=0)+p[i]-150> width:
                                    continue
                                frame[maskb[0]+q[i]+eyc-220,maskb[1]+p[i]-150]=frame3[maskb[0],maskb[1]]
                        if exc > facecx:
                            cv2.circle(frame, (exc,eyc), 15,(65, 120, 220), thickness=30, lineType=cv2.LINE_AA, shift=0)
                            cv2.circle(frame, (exc,eyc), 13,(146, 253, 255), thickness=30, lineType=cv2.LINE_AA, shift=0)
                            cv2.circle(frame, (exc,eyc), 13,(204,234,234), thickness=15, lineType=cv2.LINE_AA, shift=0)
                            cv2.line(frame, (exc, eyc), (width, eyc-120), (146, 253, 255), thickness=35, lineType=cv2.LINE_AA)
                            cv2.line(frame, (exc, eyc), (width, eyc-80), (146, 253, 255), thickness=35, lineType=cv2.LINE_AA)
                            cv2.line(frame, (exc, eyc), (width, eyc-100), (146, 253, 255), thickness=35, lineType=cv2.LINE_AA)
                            r=[]
                            l=[]
                            for j in range(5):
                                r.append((exc-width)*(1+j)//6)
                                l.append((100)*(1+j)//6)
                            r.append((exc-width)//3+20)
                            l.append((100)//3+10)
                            r.append((exc-width)//3-20)
                            l.append((100)//3-10)
                            for i in range(7):
                                if max(maskb[0],default=0)+l[i]+eyc-220> height or max(maskb[1],default=0)+r[i]-100> width:
                                    continue
                                frame[maskb[0]+l[i]+eyc-220,maskb[1]+r[i]-100]=frame3[maskb[0],maskb[1]]            
         
         
        # write the flipped frame
        out.write(frame)
        
        # check the frame size
        # print(frame.shape)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
cap2.release()
cap3.release()
out.release()
cv2.destroyAllWindows()