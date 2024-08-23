import cv2
import time
import mediapipe as mp
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
mypath = './dataset'

pTime = 0
cTime = 0

folders = [f for f in listdir(mypath)]


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


data = np.array([], dtype=float)

target = np.array([])

img = cv2.imread('1.png',1)

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

new_item = mappingHand(imgRGB)

print(new_item)

for folder in folders:
    files = [f for f in listdir(join(mypath, folder))]

    for file in files:
        img = cv2.imread(join(mypath, folder, file),1)
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        new_item = mappingHand(imgRGB) 

        data = np.vstack([data, new_item]) if data.size else np.array([new_item])
        
        target = np.append(target, folder)
        

print(data)
print(target)

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=10)

# Escolhendo um modelo de classificação - Árvore de Decisão
classifier = DecisionTreeClassifier()

# Treinando o modelo com os dados de treinamento
classifier.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred = classifier.predict(X_test)


# Avaliando o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Imprimindo os resultados
print(f"Acurácia: {accuracy:.2f}")
print("Relatório de Classificação:")
print(report)

joblib.dump(classifier, 'modelo.pkl')
