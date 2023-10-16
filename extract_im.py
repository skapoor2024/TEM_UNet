import cv2
import os

cam = cv2.VideoCapture("GB_pure_CU_400C_30minutes.mp4")

try:

    if not os.path.exists('cu_img'):
        os.makedirs('cu_img')
    
except OSError:
    print("error in the directory")

curr_frame = 0

while(True):

    ret,frame = cam.read()

    if ret:

        name = './cu_img/frame'+str(curr_frame)+'.jpg'
        print('creating..'+name)

        cv2.imwrite(name,frame)

        curr_frame+=1

    else:
        break

cam.release()
cv2.destroyAllWindows()