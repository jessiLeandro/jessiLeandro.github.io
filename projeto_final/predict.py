# JESSI LEANDRO CASTRO - 11201810509
# WELLINGTON ARAUJO DA SILVA - 11201722653
# Predição
# executar: python3 predict.py

import cv2
import mediapipe as mp
import time
import joblib
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

classifier = joblib.load('modelo.pkl')

cap = cv2.VideoCapture(0)

# Carregar os parâmetros da câmera
with np.load('parametros_calibracao.npz') as dados:
    matriz_camera = dados['matriz_camera']
    dist = dados['dist']
    rotacoes = dados['rotacoes']
    translacoes = dados['translacoes']

def mappingHand(imgRGB):
    handsMap = hands.process(imgRGB)
    result = np.array([], dtype=float)
    wrist = np.array([], dtype=float)

    if handsMap.multi_hand_landmarks:
        for handLms in handsMap.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                if id == 0:
                    wrist = np.append(result, np.array([lm.x, lm.y, lm.z], dtype=float))

                if id / 4 == 0:
                    result = np.append(result, np.array([lm.x - wrist[0] , lm.y - wrist[1], lm.z - wrist[2]], dtype=float))

                result = np.append(result, np.array([lm.x - wrist[0] , lm.y - wrist[1], lm.z - wrist[2]], dtype=float))

    return result


while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]
    new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(matriz_camera, dist, (w, h), 1, (w, h))

    # Corrigir a distorção
    img = cv2.undistort(frame, matriz_camera, dist, None, new_mtxL)    

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    pred = classifier.predict(mappingHand(imgRGB))

    cv2.putText(img, pred, (30,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


    # Press esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        print("Closing the cameras!")
        break


cap.release()
cv2.destroyAllWindows()
