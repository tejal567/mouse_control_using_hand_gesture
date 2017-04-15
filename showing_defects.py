import numpy as np
import cv2
import time
gvar=0
cap = cv2.VideoCapture(0)
start=time.time()
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
	# kernel = np.ones((3,3),np.uint8)
	# kernel2 = np.ones((3,3),np.uint8)
	# erodeout = cv2.erode(skinMask,kernel,iterations = 1)
	# dilateout = cv2.dilate(erodeout,kernel2,iterations = 1)
	# cv2.imshow("dilateoutput",dilateout)
	cv2.imshow("skinout",skinMask)
	#cannyout = cv2.Canny(dilateout,300,10)
	a,contours, hierarchy = cv2.findContours(skinMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	newim = skinMask.copy()
	newim[newim != 0]=0
	
	# faltu = []
	# faltu2 = []
	# for cnt in contours:        
	# 	area = cv2.contourArea(cnt)         
	# 	if (area > 500):
	# 		# print area
	# 		faltu.append(cnt)
	# 		hull = cv2.convexHull(cnt)
	# 		faltu2.append(hull)

	# cv2.drawContours(newim, faltu, -1, (255,255,255), 1)
	# cv2.drawContours(ghimg,faltu2,-1,(0,255,0),0)      
	# cv2.imshow("hull",ghimg)
	# cv2.imshow("contour",newim)
	
	done=time.time()
	elapsed=done-start
	# if(gvar == 0 and elapsed > 5):
	# 	gvar = 1
	# 	cnt2 = max(faltu2, key = lambda x: cv2.contourArea(x))
	# 	print cnt2
	# 	north = tuple(cnt2[cnt2[:, :, 1].argmin()][0])
	# 	print north
	# 	idx=np.where(cnt2==north)
	# 	index = np.concatenate(idx).tolist()[2]
	# 	newblack = dilateout.copy()
	# 	newblack[newblack != 0]=0
	# 	cv2.drawContours(newblack,[cnt2],-1,(255,255,255),0)
		# cv2.imshow("hullshow",newblack)
	if (elapsed>4):
		cnt = max(contours, key = lambda x: cv2.contourArea(x))
		hull = cv2.convexHull(cnt,returnPoints = False)
		if (len(hull) > 3 and len(cnt)>3):
			defects = cv2.convexityDefects(cnt,hull)
			for i in range(defects.shape[0]):
				s,e,f,d = defects[i,0]
				start1 = tuple(cnt[s][0])
				end = tuple(cnt[e][0])
				far = tuple(cnt[f][0])
				cv2.line(ghimg,start1,end,[0,255,0],2)
				cv2.circle(ghimg,far,5,[0,0,255],-1)
	cv2.imshow("frame",ghimg)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()