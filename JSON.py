import cv2
import os
import json

# Détecteur de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Créer le fichier JSON
json_file = "E:/Tous mes projets/IHM_Projects/data_JSON"

# Demander les informations de la personne
id_personne = input("Entrez l'ID de la personne : ")
nom = input("Entrez le nom : ")
prenom = input("Entrez le prénom : ")
age = input("Entrez l'âge : ")
ville = input("Entrez la ville : ")

# Créer un dossier pour stocker les images
dossier = f"E:/Tous mes projets/IHM_Projects/images/{id_personne}"
if not os.path.exists(dossier):
    os.makedirs(dossier)

# Ouvrir la webcam
cap = cv2.VideoCapture(0)
count = 0
image_principale = ""

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        image_path = f"{dossier}/{count}.jpg"
        cv2.imwrite(image_path, face)  # Enregistrer l'image

        if count == 1:
            image_principale = image_path  # Sauvegarde de l'image principale

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Capture', frame)

    if count >= 20:  # Capture 20 images
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Ajouter les infos de la personne dans JSON
person_info = {
    "nom": nom,
    "prenom": prenom,
    "age": age,
    "ville": ville,
    "image": image_principale
}

# Sauvegarder dans le fichier JSON
with open("E:/Tous mes projets/IHM_Projects/data_JSON/data.json", "w") as file:
    json.dump(person_info, file, indent=4)

print("Données et images enregistrées avec succès !")
