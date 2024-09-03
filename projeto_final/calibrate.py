# JESSI LEANDRO CASTRO - 11201810509
# WELLINGTON ARAUJO DA SILVA - 11201722653
# Calibração
# executar: python3 calibrate.py

import numpy as np 
import cv2
from tqdm import tqdm

# Set the path to the images captured by the left and right cameras
path = "./data/"

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

img_pts = []
obj_pts = []

for i in tqdm(range(1,15)):
	img = cv2.imread(path+"img%d.png"%i)
	img_gray = cv2.imread(path+"img%d.png"%i,0)

	output = img.copy()

	ret, corners = cv2.findChessboardCorners(output,(8,6),None)

	if ret:
		obj_pts.append(objp)
		cv2.cornerSubPix(img_gray, corners,(11,11),(-1,-1),criteria)
		cv2.drawChessboardCorners(output,(8,6), corners, ret)
		cv2.imshow('corners', output)
		cv2.waitKey(0)

		img_pts.append(corners)


print("Calculating camera parameters ... ")
# Calibrating  camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_gray.shape[::-1],None,None)
hL,wL= img_gray.shape[:2]
new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtx, dist, (wL,hL), 1, (wL,hL))


np.savez("parametros_calibracao.npz", matriz_camera=mtx, dist=dist, rotacoes=rvecs, translacoes=tvecs)

