from flask import Flask, render_template, request
import face_recognition
import numpy as np
import os
import cv2
import base64
from datetime import datetime
import pandas as pd

app = Flask(__name__)
KNOWN_FOLDER = 'images'
ATTENDANCE_FOLDER = 'Attendance'

os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)

# Load known faces
known_images = []
known_names = []
for filename in os.listdir(KNOWN_FOLDER):
    img = face_recognition.load_image_file(f"{KNOWN_FOLDER}/{filename}")
    known_images.append(face_recognition.face_encodings(img)[0])
    known_names.append(os.path.splitext(filename)[0])

def mark_attendance(name):
    today = datetime.now().strftime('%d-%m-%Y')
    filename = f'{ATTENDANCE_FOLDER}/Attendance_{today}.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=['Name', 'Time'])
    
    if name not in df['Name'].values:
        now = datetime.now()
        time_string = now.strftime('%H:%M:%S')
        new_entry = {'Name': name, 'Time': time_string}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(filename, index=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""
    status = ""
    if request.method == 'POST':
        image_data = request.form.get('captured_image')
        if not image_data:
            message = "No image captured!"
            status = "error"
            return render_template('index.html', message=message, status=status)
        
        # Remove header part of base64 string
        encoded_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(encoded_data)

        # Convert bytes into numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        unknown_encodings = face_recognition.face_encodings(rgb_img)

        if len(unknown_encodings) == 0:
            message = "No face detected! Try again."
            status = "error"
        else:
            unknown_encoding = unknown_encodings[0]
            matches = face_recognition.compare_faces(known_images, unknown_encoding)
            face_distances = face_recognition.face_distance(known_images, unknown_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]
                mark_attendance(name)
                message = f"Welcome {name}! Your attendance has been marked."
                status = "success"
            else:
                message = "Face not recognized! Please try again."
                status = "error"

    return render_template('index.html', message=message, status=status)

if __name__ == "__main__":
    app.run(debug=True)
