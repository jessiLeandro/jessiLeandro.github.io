# JESSI LEANDRO CASTRO - 11201810509
# WELLINGTON ARAUJO DA SILVA - 11201722653
# Caputura Imagem Para Calibração
# executar: python3 capture_images.py

import numpy as np
import cv2
import time

print("Checking the right and left camera IDs:")
print("Press (y) if IDs are correct and (n) to swap the IDs")
print("Press enter to start the process >> ")
input()

Cam= cv2.VideoCapture(0)

ret, frame= Cam.read()

cv2.imshow('img', frame)

output_path = "./data/"

start = time.time()
T = 10
count = 1

while True:
    timer = T - int(time.time() - start)
    ret, frame= Cam.read()
    
    cv2.imshow('imgL', frame)

    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray,(8,6),None)

    print("count", count)

    # If corners are detected in left and right image then we save it.
    if ret == True and timer <= 0:
        count+=1
        cv2.imwrite(output_path+'img%d.png'%count, frame)
    
    if timer <=0:
        start = time.time()
    
    # Press esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        print("Closing the cameras!")
        break

# Release the Cameras
Cam.release()
cv2.destroyAllWindows()