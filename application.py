import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import os
import zipfile
import subprocess
from flask import Flask, render_template, Response, request, jsonify, send_file, redirect, url_for, flash, session
import io
import random
import base64
from PIL import Image
import numpy as np
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


app = Flask(__name__)
app.secret_key = 'your_secret_key'


CSV_PATH = "C:/Users/nagir/Downloads/projects/Student_Identification_Track/Students_details.csv"
IMAGE_FOLDER = "C:/Users/nagir/Downloads/projects/Student_Identification_Track/Imagess"

with open("trained_data.pkl", "rb") as file:
    data = pickle.load(file)
known_encodings = data["encodings"]
known_ids = data["ids"]

student_df = pd.read_csv(CSV_PATH)
student_df.columns = student_df.columns.str.strip()

recorded_ids = set()
os.makedirs("Attendance_Records", exist_ok=True)

latest_student_info = {"name": "", "id": "", "branch": ""}
detected_student_info = {"id": None, "name": None, "branch": None}


def mark_attendance(student_id, name, branch):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    filename = f"Attendance_Records/Attendance_{date_str}.csv"

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("Timestamp,StudentID,Name,Branch\n")

    with open(filename, "a") as f:
        f.write(f"{time_str},{student_id},{name},{branch}\n")

    recorded_ids.add(student_id)


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if True in matches:
                best_match_index = face_distances.argmin()
                student_id = known_ids[best_match_index]

                student_row = student_df[student_df["StudentID"].astype(str).str.lower().str.strip() == student_id.lower().strip()]
                if not student_row.empty:
                    name = student_row.iloc[0]["Name"]
                    branch = student_row.iloc[0]["Branch"]

                    latest_student_info.update({"name": name, "id": student_id, "branch": branch})
                    detected_student_info.update({"id": student_id, "name": name, "branch": branch})

                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({student_id})", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Read users from CSV
        if not os.path.exists("users.csv"):
            return render_template('login.html', error="No users registered")

        users_df = pd.read_csv("users.csv")

        if username in users_df['username'].values:
            user_row = users_df[users_df['username'] == username].iloc[0]
            if password == user_row['password']:
                session['username'] = username
                return redirect(url_for('welcome'))
            else:
                return render_template('login.html', error="Incorrect password")
        else:
            return render_template('login.html', error="Username not found")
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            return render_template('register.html', error="All fields required")

        # Create users.csv if not exists
        if not os.path.exists("users.csv"):
            pd.DataFrame(columns=["username", "password"]).to_csv("users.csv", index=False)

        users_df = pd.read_csv("users.csv")

        if username in users_df['username'].values:
            return render_template('register.html', error="Username already exists")

        users_df.loc[len(users_df.index)] = [username, password]
        users_df.to_csv("users.csv", index=False)

        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/welcome')
@login_required
def welcome():
    return render_template('welcome.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def index():
    return render_template('index.html')


@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/latest_student')
@login_required
def latest_student():
    return jsonify(latest_student_info)


@app.route('/mark_present', methods=['POST'])
@login_required
def mark_present():
    student_id = detected_student_info.get("id")
    name = detected_student_info.get("name")
    branch = detected_student_info.get("branch")

    if student_id and name and branch:
        mark_attendance(student_id, name, branch)
        return jsonify({"status": "success", "message": "Marked Attendance"})
    else:
        return jsonify({"status": "error", "message": "No student detected."}), 400


def get_date_range(start_str, end_str):
    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
        return [start_date + pd.Timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    except:
        return []


def prepare_attendance_dataframe(start_str, end_str):
    dates = get_date_range(start_str, end_str)
    students = student_df[["StudentID", "Name", "Branch"]].copy()

    valid_columns = []
    attendance_data = {}

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        filename = f"Attendance_Records/Attendance_{date_str}.csv"
        if not os.path.exists(filename):
            continue

        try:
            df = pd.read_csv(filename)
            for session in ["Morning", "Afternoon"]:
                session_col = f"{date_str} ({session})"
                attendance_data[session_col] = [""] * len(students)
                valid_columns.append(session_col)

            for _, row in df.iterrows():
                if pd.isna(row.get("StudentID")):
                    continue
                student_id = str(row["StudentID"]).strip().lower()
                time_obj = datetime.strptime(row["Timestamp"], "%H:%M:%S")
                session = "Morning" if time_obj.hour < 12 else "Afternoon"
                col = f"{date_str} ({session})"

                match = students["StudentID"].astype(str).str.strip().str.lower() == student_id
                for idx, is_match in enumerate(match):
                    if is_match:
                        attendance_data[col][idx] = "Present"
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    # Combine attendance columns into one DataFrame
    attendance_df = pd.DataFrame(attendance_data)
    students.reset_index(drop=True, inplace=True)
    attendance_df.reset_index(drop=True, inplace=True)

    result_df = pd.concat([students, attendance_df], axis=1)
    ordered_cols = ["Name", "StudentID", "Branch"] + valid_columns
    return result_df[ordered_cols].fillna("")


@app.route('/get_attendance_data_range')
def get_attendance_data_range():
    start = request.args.get("start")
    end = request.args.get("end")
    df = prepare_attendance_dataframe(start, end)
    return jsonify(df.to_dict(orient="records"))


@app.route('/get_all_attendance')
def get_all_attendance():
    start = "2024-01-01"
    end = datetime.now().strftime("%Y-%m-%d")
    df = prepare_attendance_dataframe(start, end)
    return jsonify(df.to_dict(orient="records"))


@app.route('/calculate_attendance')
def calculate_attendance():
    student_id = request.args.get("student_id")
    start = "2024-01-01"
    end = datetime.now().strftime("%Y-%m-%d")
    df = prepare_attendance_dataframe(start, end)
    row = df[df["StudentID"].astype(str).str.lower().str.strip() == student_id.lower().strip()]

    if row.empty:
        return jsonify({"error": "Student not found"})

    present_sessions = row.iloc[0].tolist().count("Present")
    total_sessions = len([col for col in df.columns if "(" in col])
    percentage = (present_sessions / total_sessions) * 100 if total_sessions > 0 else 0

    return jsonify({"percentage": f"{percentage:.2f}"})


@app.route('/download_attendance_data_range')
def download_attendance_data_range():
    start = request.args.get("start")
    end = request.args.get("end")
    df = prepare_attendance_dataframe(start, end)

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv',
                     as_attachment=True, download_name=f"Attendance_{start}_to_{end}.csv")


@app.route('/generate_batches', methods=['GET'])
def generate_batches():
    size = request.args.get('size', default=5, type=int)
    students = student_df.to_dict(orient='records')
    random.shuffle(students)
    batches = [students[i:i + size] for i in range(0, len(students), size)]
    return jsonify(batches)


@app.route('/upload_data', methods=['GET', 'POST'])
@login_required
def upload_data():
    if request.method == 'POST':
        if 'csv_file' not in request.files or 'zip_file' not in request.files:
            flash('Both CSV and ZIP files are required.', 'error')
            return redirect(request.url)

        csv_file = request.files['csv_file']
        zip_file = request.files['zip_file']

        if csv_file and csv_file.filename.endswith('.csv'):
            csv_file.save(CSV_PATH)

        if zip_file and zip_file.filename.endswith('.zip'):
            with zipfile.ZipFile(zip_file) as zip_ref:
                zip_ref.extractall(IMAGE_FOLDER)

        subprocess.run(["python", "train_faces.py"], check=True)
        flash(f"Uploaded: {csv_file.filename}, {zip_file.filename}", "success")
        return redirect(url_for('index'))

    return render_template("upload.html")

@app.route('/add_student', methods=['GET', 'POST'])
@login_required
def add_student():
    if request.method == 'POST':
        name = request.form.get("name").strip()
        student_id = request.form.get("student_id").strip()
        branch = request.form.get("branch").strip()
        image = request.files.get("image")

        if not (name and student_id and branch and image):
            return "All fields required", 400

        # Save image
        image_path = os.path.join(IMAGE_FOLDER, f"{student_id}.jpg")
        image.save(image_path)

        # Add to CSV if student_id not already present
        df = pd.read_csv(CSV_PATH)
        if student_id in df["StudentID"].astype(str).values:
            return "Student ID already exists.", 400

        df.loc[len(df.index)] = [name, student_id, branch]
        df.to_csv(CSV_PATH, index=False)

        # Retrain encodings only for the new face
        subprocess.run(["python", "train_faces.py"], check=True)

        return redirect(url_for('welcome'))

    return render_template("add_student.html")

@app.route('/add_student_tab', methods=['POST'])
@login_required
def add_student_tab():
    name = request.form.get("name").strip()
    student_id = request.form.get("student_id").strip()
    branch = request.form.get("branch").strip()
    image = request.files.get("image")

    if not (name and student_id and branch and image):
        flash("All fields are required.", "error")
        return redirect(url_for('index', tab='add_student'))

    image_path = os.path.join(IMAGE_FOLDER, f"{student_id}.jpg")
    image.save(image_path)

    df = pd.read_csv(CSV_PATH)
    if student_id in df["StudentID"].astype(str).values:
        flash("Student ID already exists.", "error")
        return redirect(url_for('index', tab='add_student'))

    df.loc[len(df.index)] = [name, student_id, branch]
    df.to_csv(CSV_PATH, index=False)

    subprocess.run(["python", "train_faces.py"], check=True)
    flash("Student added successfully!", "success")
    return redirect(url_for('index', tab='add_student'))

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'message': 'No image provided'})

    # Convert base64 image to OpenCV frame
    image_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Face recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            student_id = known_ids[best_match_index]
            student_row = student_df[student_df["StudentID"].astype(str).str.lower().str.strip() == student_id.lower().strip()]

            if not student_row.empty:
                name = student_row.iloc[0]["Name"]
                branch = student_row.iloc[0]["Branch"]
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Save to attendance file
                date_str = datetime.now().strftime('%Y-%m-%d')
                filename = f"Attendance_Records/Attendance_{date_str}.csv"

                if os.path.exists(filename):
                    existing = pd.read_csv(filename)
                else:
                    existing = pd.DataFrame(columns=["Timestamp", "StudentID", "Name", "Branch"])

                already_marked = (
                    (existing['StudentID'].astype(str).str.lower() == student_id.lower()) &
                    (existing['Timestamp'].str[:10] == now[:10])
                ).any()

                if not already_marked:
                    new_row = {
                        "Timestamp": now,
                        "StudentID": student_id,
                        "Name": name,
                        "Branch": branch
                    }
                    existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
                    existing.to_csv(filename, index=False)

                return jsonify({'message': f"Attendance marked for {name}"})

    return jsonify({'message': 'Face not recognized'})

application = app


if __name__ == '__main__':

    app.run(debug=True)

