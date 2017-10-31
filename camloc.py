#camera visualization
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
 #           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

img_rgb = cv2.imread('Images/IMG_6719.JPG')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('Images/pattern.png',0)
w, h = template.shape[::-1]
#method = eval(meth)
res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)
threshold = 0.9
loc= np.where(res >=threshold)

for pt in zip(*loc[::-1]):
	cv2.rectangle(img_rgb,pt, (pt[0]+w,pt[1]+h), (0,255,255), 2)

plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB),  interpolation='bicubic')
#plt.plot([50,100],[80,100], 'c', linewidth=5)
plt.show()
"""
#trial
img1 = cv2.imread('Images/pattern.png',0) # queryImage
img_rgb = cv2.imread('Images/IMG_6719.JPG')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img2 = cv2.resize(img_gray,None,fx=0.35, fy=0.35, interpolation = cv2.INTER_LINEAR )

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)

plt.imshow(img3)
plt.show()
