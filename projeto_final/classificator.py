# JESSI LEANDRO CASTRO - 11201810509
# WELLINGTON ARAUJO DA SILVA - 11201722653
# Treinamento
# executar: python3 capture2dataset.py

import cv2
import time
import mediapipe as mp
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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

for folder in folders:
    files = [f for f in listdir(join(mypath, folder))]

    for file in files:
        img = cv2.imread(join(mypath, folder, file),1)
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        new_item = mappingHand(imgRGB)

        if new_item.size == 0:
            continue

        data = np.vstack([data, new_item]) if data.size else np.array([new_item])
        
        target = np.append(target, folder)
        

print(data)
print(target)

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=10)

# Escolhendo um modelo de classificação - Árvore de Decisão
# classifier = DecisionTreeClassifier()
# classifier = RandomForestClassifier()
classifier = KNeighborsClassifier()

# Treinando o modelo com os dados de treinamento
classifier.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred = classifier.predict(X_test)


# Avaliando o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Imprimindo os resultados
print(f"Acurácia: {accuracy:.2f}")
print("Relatório de Classificação:")
print(report)

joblib.dump(classifier, 'modelo.pkl')


# Extrair as métricas (Precisão, Recall, F1-score) para cada classe
precision = [v['precision'] for k, v in report.items() if k != 'accuracy' and k != 'macro avg' and k != 'weighted avg']
recall = [v['recall'] for k, v in report.items() if k != 'accuracy' and k != 'macro avg' and k != 'weighted avg']
f1_score = [v['f1-score'] for k, v in report.items() if k != 'accuracy' and k != 'macro avg' and k != 'weighted avg']

# Classes
classes = [k for k in report.keys() if k != 'accuracy' and k != 'macro avg' and k != 'weighted avg']

print(classes)

# Configurar o gráfico
x = np.arange(len(classes))  # Localização das classes no eixo x
width = 0.2  # Largura das barras

fig, ax = plt.subplots(figsize=(10, 6))

# Plotar as barras
bars1 = ax.bar(x - width, precision, width, label='Precisão', color='skyblue')
bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='lightcoral')

# Adicionar as legendas e título
ax.set_xlabel('Classes')
ax.set_ylabel('Valor')
ax.set_title('Métricas de Desempenho por Classe')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Adicionar os valores das métricas em cima das barras
def autolabel(bars):
    """Adiciona rótulos nas barras"""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 pontos de deslocamento acima da barra
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

plt.show()


# Calculando métricas
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Matriz de Confusão')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.show()