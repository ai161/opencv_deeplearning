import numpy as np
import cv2

cap1 = cv2.VideoCapture('output.mp4')
cap2 = cv2.VideoCapture('泡.gif')

while(cap1.isOpened()):
    ret, frame = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret2:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    if ret:
       mask = np.where(((frame2!=[195,195,195]) & (frame2!=[194,194,194])).all(axis=2))
       frame[mask]=frame2[mask]
       cv2.imshow('frame',frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()


