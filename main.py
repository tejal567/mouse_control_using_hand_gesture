import cv2
import math
import sys
import numpy as np
import pyautogui
import time


#cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier('face.xml')

video_capture = cv2.VideoCapture(0)
pyautogui.FAILSAFE =  False
ptx =0
pty =0
t1 = time.time()
while True:
    # Capture frame-by-frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    face_detect = False
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    ghimg = frame.copy()
    ghimg1 = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original",ghimg1)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x-20, y-20), (x+w+10, y+h+50), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, y+h+100), (1000, 1000), (0, 0, 0), -1)
        face_detect = True

    # Display the resulting frame
    if face_detect == False:
        continue

    lower = np.array([0, 48, 80], dtype = "uint8")
    #lower = np.array([140, 100, 40], dtype = "uint8")
    upper = np.array([25, 255, 255], dtype = "uint8")
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    kernel = np.ones((5,5),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    erodeout = cv2.erode(skinMask,kernel,iterations = 1)
    dilateout = cv2.dilate(erodeout,kernel2,iterations = 2)
    cv2.imshow("dilate_output",dilateout)

   



    _,contours, hierarchy = cv2.findContours(dilateout,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    newim = dilateout.copy()
    newim[newim != 0]=0

    faltu = []
    faltu2 = []
    for cnt2 in contours:        
        area = cv2.contourArea(cnt2)         
        if (area > 1000):
            # print area
            north = tuple(cnt2[cnt2[:, :, 1].argmin()][0])
            # if(north[1] > 300):
            #     continue
            #print north
            idx=np.where(cnt2==north)
            index = np.concatenate(idx).tolist()[2]
            length = len(cnt2)
            rightindex = -1
            leftindex = -1
            remove_contour = 0
            count = 0
            for i in range(0,length):
                if (abs(north[1] - cnt2[i][0][1])<20):
                    count += 1
            if count > 20:
                continue

            #print cnt2[1][0]
            for i in range (index+1,length):
                #print cnt2[i][0][0],cnt2[i][0][1]
                if (abs(cnt2[i][0][1] - north[1]) < 80 and abs(cnt2[i][0][0] - north[0]) < 80):
                    continue
                else :
                    rightindex = i
                    break

            if rightindex == -1:
                for i in range (0 , index):
                    if (abs(cnt2[i][0][1] - north[1]) < 80 and abs(cnt2[i][0][0] - north[0]) < 80):
                        continue
                    else :
                        rightindex = i
                        break
            for i in range (0,index):
                if (abs(cnt2[index-1-i][0][1] - north[1]) < 80 and abs(cnt2[index-1-i][0][0] - north[0]) < 80):
                    continue
                else :
                    leftindex = index-1-i
                    break

            if leftindex == -1:
                for i in range (index+1 , length):
                    if (abs(cnt2[length+index-i][0][1] - north[1]) < 80 and abs(cnt2[length+index-i][0][0] - north[0]) < 80):
                        continue
                    else :
                        leftindex = length+index-i
                        break
            #print leftindex,rightindex
            if leftindex == -1 or rightindex == -1 or leftindex == rightindex:
                remove_contour = 1
            #print cnt2[leftindex][0]
            #print cnt2[rightindex][0]
            if remove_contour == 0:
                sidea = math.sqrt(((cnt2[leftindex][0][0]-cnt2[rightindex][0][0])*(cnt2[leftindex][0][0]-cnt2[rightindex][0][0]))+((cnt2[leftindex][0][1]-cnt2[rightindex][0][1])*(cnt2[leftindex][0][1]-cnt2[rightindex][0][1])))
                sideb = math.sqrt(((cnt2[index][0][0]-cnt2[rightindex][0][0])*(cnt2[index][0][0]-cnt2[rightindex][0][0]))+((cnt2[index][0][1]-cnt2[rightindex][0][1])*(cnt2[index][0][1]-cnt2[rightindex][0][1])))
                sidec = math.sqrt(((cnt2[leftindex][0][0]-cnt2[index][0][0])*(cnt2[leftindex][0][0]-cnt2[index][0][0]))+((cnt2[leftindex][0][1]-cnt2[index][0][1])*(cnt2[leftindex][0][1]-cnt2[index][0][1])))
                #print sidea,sideb,sidec
                cos = ((sidec*sidec)+(sideb*sideb)-(sidea*sidea))/(2*sidec*sideb)
                if cos>1:
                    continue
                angle = (180/math.pi)*math.acos(cos)
                if angle < 80:
                    faltu.append(cnt2)
                    hull = cv2.convexHull(cnt2)
                    faltu2.append(hull)

    if (len(faltu) == 0):
        continue
    cnt_max = max(faltu,key = lambda x:cv2.contourArea(x))
    hull_max = cv2.convexHull(cnt_max)
    topmost = tuple(cnt_max[cnt_max[:, :, 1].argmin()][0])
    if abs(pty - topmost[1]) > 70 or abs(ptx - topmost[0]) > 70:
        pty = topmost[1]
        ptx = topmost[0]
        continue
    t2 = time.time()
    distance = math.sqrt(math.pow(pty - topmost[1],2)+math.pow(ptx - topmost[0],2))
    if t2 - t1 < 1.0 and distance < 20.0:
        ghi = 1
    else:
        pty = topmost[1]
        ptx = topmost[0]
        pyautogui.moveTo(ptx*1366/640,pty*768/480,0.2)
    t1 = t2
    cv2.drawContours(newim, [cnt_max], -1, (255,255,255), 1)
    cv2.drawContours(ghimg,[hull_max],-1,(0,255,0),0)      
    cv2.imshow("hull",ghimg)
    cv2.imshow("contour",newim)

    
    

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

