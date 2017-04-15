#without using face cascade

import numpy as np
import cv2
import time
import pyautogui
gvar=0
cap = cv2.VideoCapture(0)
start=time.time()
ptx =0
pty =0
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	#for horizontal invert
	fimg=cv2.flip(frame,1)
	fimg = cv2.GaussianBlur(fimg,(5,5),0)
	ghimg = fimg.copy()
	lower = np.array([0, 48, 80], dtype = "uint8")
	upper = np.array([25, 255, 255], dtype = "uint8")
	converted = cv2.cvtColor(fimg, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(converted, lower, upper)
	kernel = np.ones((3,3),np.uint8)
	kernel2 = np.ones((3,3),np.uint8)
	erodeout = cv2.erode(skinMask,kernel,iterations = 1)
	dilateout = cv2.dilate(erodeout,kernel2,iterations = 1)
	cv2.imshow("dilateoutput",dilateout)
	#cannyout = cv2.Canny(dilateout,300,10)
	_,contours, hierarchy = cv2.findContours(skinMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	newim = dilateout.copy()
	newim[newim != 0]=0
	blackscreen=newim.copy()
	faltu = []
	faltu2 = []
	for cnt in contours:        
		area = cv2.contourArea(cnt)         
		if (area > 500):
			# print area
			faltu.append(cnt)
			hull = cv2.convexHull(cnt)
			faltu2.append(hull)

	cv2.drawContours(newim, faltu, -1, (255,255,255), 1)
	cv2.drawContours(ghimg,faltu2,-1,(0,255,0),0)      
	cv2.imshow("hull",ghimg)
	cv2.imshow("contour",newim)
	
	done=time.time()
	elapsed=done-start
	if(gvar == 0 and elapsed > 5):
		# gvar = 1
		# cnt2 = max(faltu2, key = lambda x: cv2.contourArea(x))
		# print cnt2
		# north = tuple(cnt2[cnt2[:, :, 1].argmin()][0])
		# print north
		# idx=np.where(cnt2==north)
		# index = np.concatenate(idx).tolist()[2]
		# newblack = dilateout.copy()
		# newblack[newblack != 0]=0
		# cv2.drawContours(newblack,[cnt2],-1,(255,255,255),0)

		cnt_max = max(faltu,key = lambda x:cv2.contourArea(x))
		hull_max = cv2.convexHull(cnt_max)
		
		#showing maximum area contour
		cv2.drawContours(blackscreen,cnt_max,-1,(255,255,255), 1)
		cv2.imshow("max contour",blackscreen)
		topmost = tuple(cnt_max[cnt_max[:, :, 1].argmin()][0])
		if pty - topmost[1] > 70 or ptx - topmost[0] > 70:
			pty = topmost[1]
			ptx = topmost[0]
			continue
		pty = topmost[1]
		ptx = topmost[0]
		pyautogui.moveTo(ptx*1366/640,pty*768/480)
		# cv2.imshow("hullshow",newblack)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()