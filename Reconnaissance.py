import cv2
import json
import numpy as np

# Charger le fichier JSON
json_file = "E:/Tous mes projets/IHM_Projects/data_JSON/data.json"
with open(json_file, "r") as file:
    data = json.load(file)

# Charger le modèle entraîné
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")

# Détecteur de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        id, confidence = recognizer.predict(face)

        if confidence < 100:  # Seuil de reconnaissance
            if str(id) in data:
                personne = data[str(id)]
                nom, prenom, age, ville = personne["nom"], personne["prenom"], personne["age"], personne["ville"]
                images = personne["images"]

                text = f"{nom} {prenom}, {age} ans, {ville}"
                color = (0, 255, 0)

                # Charger et afficher toutes les images de la personne reconnue
                for image in images:
                    img = cv2.imread('E:/Tous mes projets/IHM_Projects/images/1/1.jpg')
                    if img is not None:
                        cv2.imshow(f"Photos de {nom}", img)

            else:
                text = "Inconnu"
                color = (0, 0, 255)

        else:
            text = "Inconnu"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Reconnaissance faciale', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
