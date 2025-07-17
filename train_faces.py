import os
import cv2
import face_recognition
import pickle

data_path = "C:/Users/nagir/Downloads/projects/Student_Identification_Track/Imagess"
known_encodings = []
known_ids = []

print("[INFO] Training started...")

for filename in os.listdir(data_path):
    if filename.lower().endswith((".jpg", ".png")):
        img_path = os.path.join(data_path, filename)
        image = cv2.imread(img_path)

        # Resize to speed up face detection
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes = face_recognition.face_locations(rgb_image, model="hog")
        encodings = face_recognition.face_encodings(rgb_image, boxes)

        if encodings:
            known_encodings.append(encodings[0])
            student_id = os.path.splitext(filename)[0]
            known_ids.append(student_id)
            print(f"[INFO] Processed {filename} → ID: {student_id}")
        else:
            print(f"[WARNING] No face found in {filename}")

# Save to pickle file
data = {"encodings": known_encodings, "ids": known_ids}
with open("trained_data.pkl", "wb") as file:
    pickle.dump(data, file)

print("[INFO] Training completed and data saved to 'trained_data.pkl'")
