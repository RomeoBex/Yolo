from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
import matplotlib.pyplot as plt

# Charger le modèle YOLO
model = YOLO("yolov8n.pt")

# Charger l'image à partir d'une URL
response = requests.get("https://img.freepik.com/vecteurs-premium/transport-pour-voyage-voiture-train-bus-croiseur-avion_104045-4125.jpg")
image = Image.open(BytesIO(response.content))
image = np.asarray(image)

# Créer une copie mutable de l'image (résolution du problème de lecture seule)
image = np.array(image, dtype=np.uint8).copy()

# Prédire avec YOLO 
results = model.predict(image)

# Fonction pour dessiner les boîtes et les labels
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # épaisseur du texte
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # largeur et hauteur du texte
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # remplissage
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

# Fonction pour dessiner toutes les boîtes sur l'image
def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
    # Définir les labels COCO si non spécifiés
    if labels == []:
        labels = {0: u'__background__', 1: u'person', 2: u'bicycle', 3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog', 54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'}
    
    # Définir les couleurs si non spécifiés
    if colors == []:
        colors = [(89, 161, 197), (67, 161, 255), (19, 222, 24), (186, 55, 2), (167, 146, 11), (190, 76, 98), (130, 172, 179)]
  
    # Dessiner chaque boîte sur l'image
    for box in boxes:
        if score:
            label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]), 1)) + "%"
        else:
            label = labels[int(box[-1])+1]
        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1]) % len(colors)]
                box_label(image, box, label, color)
        else:
            color = colors[int(box[-1]) % len(colors)]
            box_label(image, box, label, color)

    # Convertir l'image BGR en RGB pour l'affichage avec matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Appeler la fonction pour tracer les boîtes
plot_bboxes(image, results[0].boxes.data, score=False)

# Affichage de l'image avec matplotlib (utile pour les environnements sans interface graphique)
plt.imshow(image)
plt.axis('off')  # Masquer les axes
plt.show()
