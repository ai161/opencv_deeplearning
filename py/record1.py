import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('xml/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('xml/haarcascade_mcs_nose.xml')


cap = cv2.VideoCapture(0)
cap4 = cv2.VideoCapture('hartt.gif')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (1280,720))

num=1

while(cap.isOpened()):
    ret, frame = cap.read()
    ret4, frame4 = cap4.read()
    if not ret4:
        cap4.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        maskh = np.where((frame4!=[255,255,300]).all(axis=2))
        

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            
            dst = frame[y:y + h, x:x + w]
            save_path =  'out' + '/'  + str(num) + '.jpg'
            a = cv2.imwrite(save_path, dst)
            num+=1
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            height, width, channels =frame.shape[:3]
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            facecx=x+(w/2)
            eyes = eye_cascade.detectMultiScale(roi_gray) 
            noses=nose_cascade.detectMultiScale(roi_gray) 
            counte=0
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                counte=counte+1
            if counte ==2:
                for (ex,ey,ew,eh) in eyes:
                    exc= int(ex + (ew/2))+x
                    eyc= int(ey + (eh/2))+y
                    if exc <= facecx:
                        cv2.line(frame, (exc, eyc), (0,eyc-100), (152, 145, 234), thickness=10, lineType=cv2.LINE_AA)
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
out.release()
cv2.destroyAllWindows()