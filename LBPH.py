import cv2
import numpy as np
import os

DATASET_DIR = "E:/Tous mes projets/IHM_Projects/dataset"
MODEL_FILE = "face_recognizer.yml"

# Initialisation du modèle de reconnaissance faciale LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Chargement du détecteur de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Listes pour stocker les images et les labels
faces = []
labels = []

# Parcourir les dossiers des personnes
for person_id in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_id)
    
    if not os.path.isdir(person_folder):
        continue
    
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        
        # Charger l'image en niveaux de gris
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        faces.append(img)
        labels.append(int(person_id))  # L'ID est converti en entier pour OpenCV

# Conversion en format numpy
faces = np.array(faces, dtype="object")
labels = np.array(labels)

# Entraîner le modèle LBPH
print("Entraînement du modèle en cours...")
recognizer.train(faces, labels)
print("Modèle entraîné avec succès !")

# Sauvegarde du modèle
recognizer.save(MODEL_FILE)
print(f"Modèle enregistré sous '{MODEL_FILE}'")
