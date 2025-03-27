import cv2
import os
# Création du dossier pour stocker les images
DATASET_DIR = '"E:/Tous mes projets/IHM_Projects/dataset"'

# Demander un identifiant unique pour la personne
person_id = input("Entrez un ID unique pour la personne : ")
person_folder = os.path.join(DATASET_DIR, person_id)

# Initialisation de la webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
max_images = 50  # Nombre d'images à capturer

while count < max_images:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        img_path = os.path.join(person_folder, f"{count}.jpg")
        cv2.imwrite(img_path, face)
        count += 1
        
        # Dessiner un rectangle autour du visage
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Capture de Visage", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Capture terminée. {count} images enregistrées dans {person_folder}")
