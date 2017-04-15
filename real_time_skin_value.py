import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
start=time.time()
var=False
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#for horizontal invert
	fimg=cv2.flip(frame,1)
	ghimg = fimg.copy()
	# Display the resulting frame
	cv2.rectangle(fimg,(150,150),(165,165),(0,255,0),1)
	cv2.rectangle(fimg,(250,150),(265,165),(0,255,0),1)
	cv2.rectangle(fimg,(150,300),(165,315),(0,255,0),1)
	cv2.rectangle(fimg,(250,300),(265,315),(0,255,0),1)
	cv2.rectangle(fimg,(200,250),(215,265),(0,255,0),1)
	# cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	
	cv2.imshow('frame',fimg)
	done=time.time()
	elapsed=done-start
	# print elapsed
	# rmin=255
	# rmax=0
	# gmin=255
	# gmax=0
	# bmin=255
	# bmax=0
	ravg = 0
	bavg = 0
	gavg = 0

	if(elapsed>=5 and var==False):
		for x in range(155,160):
			for y in range(155,160):
				# if(bmax<fimg[x][y][0]):
				# 	bmax=fimg[x][y][0]
				# if(bmin>fimg[x][y][0]):
				# 	bmin=fimg[x][y][0]
				# if(gmax<fimg[x][y][1]):
				# 	gmax=fimg[x][y][1]
				# if(gmin>fimg[x][y][1]):
				# 	gmin=fimg[x][y][1]
				# if(rmax<fimg[x][y][2]):
				# 	rmax=fimg[x][y][2]
				# if(rmin>fimg[x][y][2]):
				# 	rmin=fimg[x][y][2]
				ravg += fimg[x][y][2]
				gavg += fimg[x][y][1]
				bavg += fimg[x][y][0]
		ravg /= 25
		gavg /= 25
		bavg /= 25

		for x in range(255,260):
			for y in range(155,160):
				
				ravg += fimg[x][y][2]
				gavg += fimg[x][y][1]
				bavg += fimg[x][y][0]
		ravg /= 25
		gavg /= 25
		bavg /= 25
		
		for x in range(155,160):
			for y in range(305,310):
				ravg += fimg[x][y][2]
				gavg += fimg[x][y][1]
				bavg += fimg[x][y][0]
		ravg /= 25
		gavg /= 25
		bavg /= 25
		for x in range(255,260):
			for y in range(305,310):
				ravg += fimg[x][y][2]
				gavg += fimg[x][y][1]
				bavg += fimg[x][y][0]
		ravg /= 25
		gavg /= 25
		bavg /= 25
		for x in range(205,210):
			for y in range(255,260):
				ravg += fimg[x][y][2]
				gavg += fimg[x][y][1]
				bavg += fimg[x][y][0]
		ravg /= 25
		gavg /= 25
		bavg /= 25
		var=True
		print (bavg,gavg,ravg)

	if(var == True):
		lower = np.array([0, 48, 80], dtype = "uint8")
		upper = np.array([20, 255, 255], dtype = "uint8")
		converted = cv2.cvtColor(fimg, cv2.COLOR_BGR2HSV)
		skinMask = cv2.inRange(converted, lower, upper)
		kernel = np.ones((3,3),np.uint8)
		# skinMask = cv2.erode(skinMask,kernel,iterations = 1)
		skinMask = cv2.dilate(skinMask,kernel,iterations = 1)
		cv2.imshow("img",skinMask)
		# skinMask = cv2.Canny(skinMask,100,200)
		testr,contours, hierarchy = cv2.findContours(skinMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(skinMask, contours, 0, (255,255,255), 3)
		cnt = max(contours, key = lambda x: cv2.contourArea(x))
		hull = cv2.convexHull(cnt)
		cv2.drawContours(ghimg,[hull],0,(0,255,0),0)
		cv2.imshow('frame1',ghimg)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()