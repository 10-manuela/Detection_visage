import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk

# Fonction pour afficher l'image dans Tkinter
def show_image(frame, panel):
    img = Image.fromarray(frame)
    img = img.resize((640, 480), Image.ANTIALIAS)  # Redimensionner pour afficher
    img = ImageTk.PhotoImage(img)
    
    panel.configure(image=img)
    panel.image = img  # Garder une référence pour éviter que l'image soit supprimée

# Fonction pour démarrer la vidéo et la reconnaissance faciale
def start_video():
    cap = cv2.VideoCapture(0)  # Utiliser la caméra
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            id, confidence = recognizer.predict(face)
            if confidence < 100:
                # Chercher les informations dans la base de données
                query = "SELECT * FROM personnes WHERE id = %s"
                cursor.execute(query, (id,))
                result = cursor.fetchone()
                if result:
                    update_person_info(result)  # Mettre à jour les informations dans l'UI
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Dessiner un rectangle autour du visage
        
        # Afficher l'image dans Tkinter
        show_image(frame, panel)
        window.update()

# Fonction pour mettre à jour les informations de la personne dans l'UI
def update_person_info(info):
    name_label.config(text=f"Nom: {info[1]} {info[2]}")
    age_label.config(text=f"Âge: {info[3]}")
    city_label.config(text=f"Ville: {info[4]}")
    level_label.config(text=f"Niveau d'étude: {info[5]}")

# Créer la fenêtre principale Tkinter
window = tk.Tk()
window.title("Application de Reconnaissance Faciale")

# Créer un cadre pour afficher l'image vidéo
panel = tk.Label(window)
panel.grid(row=0, column=0, padx=10, pady=10)

# Créer des labels pour afficher les informations de la personne reconnue
info_frame = tk.Frame(window)
info_frame.grid(row=0, column=1, padx=10, pady=10)

name_label = tk.Label(info_frame, text="Nom: ", font=("Arial", 12))
name_label.grid(row=0, column=0, pady=5)
age_label = tk.Label(info_frame, text="Âge: ", font=("Arial", 12))
age_label.grid(row=1, column=0, pady=5)
city_label = tk.Label(info_frame, text="Ville: ", font=("Arial", 12))
city_label.grid(row=2, column=0, pady=5)
level_label = tk.Label(info_frame, text="Niveau d'étude: ", font=("Arial", 12))
level_label.grid(row=3, column=0, pady=5)

# Créer un bouton pour démarrer la vidéo et la reconnaissance faciale
start_video_button = tk.Button(window, text="Démarrer la reconnaissance", command=start_video, font=("Arial", 14))
start_video_button.grid(row=1, column=0, columnspan=2, pady=10)

# Ajouter un bouton de fermeture
close_button = tk.Button(window, text="Fermer", command=window.quit, font=("Arial", 14))
close_button.grid(row=2, column=0, columnspan=2, pady=10)

# Charger le détecteur de visages et le modèle de reconnaissance
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('E:/Tous mes projets/IHM_Projects/face_recognizer.yml')

# Connexion à la base de données
import mysql.connector
conn = mysql.connector.connect(host="localhost", user="root", password="ton_mot_de_passe", database="base_reconnaissance")
cursor = conn.cursor()

# Lancer l'application
window.mainloop()
