#camera visualization
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

MIN_MATCH_COUNT = 10
VALID_NORM = 1e-6
FOCAL_WIDTH = 330.0
FOCAL_HEIGHT = 330.0

def images(queryImage,trainImage):
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)


	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)
	good_matches  = []
	for m, n in matches:
		if m.distance < 0.5 * n.distance:
			good.append([m])

	pattern_pts = []
	desired_img_pts = []
	n = min(len(kp1), len(kp2))
	for i in range(n):
		pattern_pts.append(kp1[i].pt)
		desired_img_pts.append(kp2[i].pt)
	pattern_pts=np.float32(pattern_pts)
	desired_img_pts= np.float32(desired_img_pts)
	# Obtain the camera matrix for the desired img
	width, height = trainImage.shape[:3]

	fx = FOCAL_WIDTH
	fy = FOCAL_HEIGHT
	cx = width / 2
	cy = height / 2

	camera_mat = np.float32([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

	# Obtain the essential matrix, which will be used to get the estimated pose matrix

	essential_mat, essential_mat_mask = cv2.findEssentialMat(pattern_pts, desired_img_pts,1.0, (cx,cy), cv2.RANSAC, 0.999, 3,mask)

	# Obtain the estimated pose by using the essential matrix (similar to the homography matrix)
	pose_mat = cv2.recoverPose(essential_mat, src_pts, dst_pts, camera_mat)

	c, rotation_mat, translation_mat= pose_mat[:3]

	euler_angles_mat = rotationMatrixToEulerAngles(rotation_mat)
	return euler_angles_mat,desired_img_pts


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

	assert(isRotationMatrix(R))

	sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

	singular = sy < 1e-6

	if  not singular :
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0

	return np.array([x, y, z])

#get angle from homography matrix
def getComponents(hg):
	a=hg[0,0]
	b=hg[0,1]
	theta = np.arctan2(b,a)*(180/np.pi)
	return theta

def rotate_bound(image, angle):
	# grab the dimensions of the image and then determine the
	# center
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)

	# grab the rotation matrix (applying the negative of the
	# angle to rotate clockwise), then grab the sine and cosine
	# (i.e., the rotation components of the matrix)
	M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	# compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))

	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	# perform the actual rotation and return the image
	return cv2.warpAffine(image, M, (nW, nH))




queryImage = cv2.imread('Images/pattern.png',0) # queryImage
img_rgb = cv2.imread('Images/IMG_6720.JPG') #trainImage
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
trainImage = cv2.resize(img_gray,None,fx=0.35, fy=0.35, interpolation = cv2.INTER_LINEAR )
a,dst= images(queryImage,trainImage)
print "roll: ",a[0],"pitch: ",a[1],"yaw: ",a[2]
#repeat for each image

#thetha=getComponents(M)

#print "Angle: ",thetha,"Train image pos: ",dst
#thetha_rad=thetha*(np.pi/180)
#imgtop=rotate_bound(img2, thetha)
#plt.imshow(img3, 'gray'),plt.show()
#plt.imshow(imgtop, 'gray'),plt.show()

#tried BFtemplate matching
"""
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
"""
