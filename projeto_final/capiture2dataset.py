import cv2
import numpy as np 

Cam = cv2.VideoCapture(0)

ret, frame = Cam.read()

cv2.imshow('img', frame)

output_path = "./Dataset/"

count = 0

# Carregar os parâmetros da câmera
with np.load('parametros_calibracao.npz') as dados:
    matriz_camera = dados['matriz_camera']
    dist = dados['dist']
    rotacoes = dados['rotacoes']
    translacoes = dados['translacoes']

while True:
    ret, frame = Cam.read()
    
    cv2.imshow('img', frame)

    h, w = frame.shape[:2]
    new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(matriz_camera, dist, (w, h), 1, (w, h))

    # Corrigir a distorção
    img = cv2.undistort(frame, matriz_camera, dist, None, new_mtxL)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        count+=1
        cv2.imwrite(output_path+'A/img%d.png'%count, img)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        count+=1
        cv2.imwrite(output_path+'E/img%d.png'%count, img)

    if cv2.waitKey(1) & 0xFF == ord('i'):
        count+=1
        cv2.imwrite(output_path+'I/img%d.png'%count, img)

    if cv2.waitKey(1) & 0xFF == ord('o'):
        count+=1
        cv2.imwrite(output_path+'O/img%d.png'%count, img)

    if cv2.waitKey(1) & 0xFF == ord('u'):
        count+=1
        cv2.imwrite(output_path+'U/img%d.png'%count, img)

    # Press esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        print("Closing the cameras!")
        break

# Release the Cameras
Cam.release()
cv2.destroyAllWindows()