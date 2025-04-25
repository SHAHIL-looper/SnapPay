import cv2
import os
import numpy as np
from datetime import datetime
import pyttsx3

# ========== SETUP ==========
os.makedirs("dataset", exist_ok=True)
os.makedirs("Employee", exist_ok=True)

# ========== VOICE ==========
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak(text):
    print("[VOICE] " + text)
    engine.say(text)
    engine.runAndWait()

# ========== UTILITY ==========
def print_banner():
    print("\n" + "=" * 40)
    print("     FACE ATTENDANCE SYSTEM")
    print("=" * 40)

def student_exists(student_id):
    if not os.path.exists("Employee/info.txt"):
        return False
    with open("Employee/info.txt", "r") as f:
        return any(line.startswith(str(student_id) + ",") for line in f)

def load_students():
    student_dict = {}
    if os.path.exists("Employee/info.txt"):
        with open("Employee/info.txt", "r") as f:
            for line in f:
                sid, name = line.strip().split(",")
                student_dict[int(sid)] = name
    return student_dict

def save_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    already_marked = False

    if os.path.exists("attendance.csv"):
        with open("attendance.csv", "r") as f:
            already_marked = any(name in line and today in line for line in f)

    if not already_marked:
        with open("attendance.csv", "a") as f:
            f.write(f"{name},{timestamp}\n")
        print(f"[✓] Attendance marked for {name} at {timestamp}")
        speak(f"Attendance marked for {name}")

# ========== CORE FUNCTIONS ==========
def register_face():
    print("\n[REGISTER EMPLOYEE]")
    try:
        student_id = int(input("Enter Employee ID: "))
    except ValueError:
        print("[!] Invalid ID. Must be a number.")
        return

    student_name = input("Enter Employee name: ").strip()
    if not student_name:
        print("[!] Name cannot be empty.")
        return

    if student_exists(student_id):
        print(f"[!] Employee ID {student_id} is already registered.")
        speak(f"Employee ID {student_id} is already registered.")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    with open("Employee/info.txt", "a") as f:
        f.write(f"{student_id},{student_name}\n")

    print(f"[INFO] Capturing face data for {student_name}... Look at the camera.")
    speak(f"Capturing face data for {student_name}")

    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Failed to access the camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"dataset/User.{student_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Registering - Press Q to Quit", img)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[✓] Registered {student_name} with {count} face samples.")
    speak(f"Registration complete for {student_name}")

def train_model():
    print("\n[TRAINING MODEL]")
    speak("Training the face recognition model now.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []

    for file in os.listdir("dataset"):
        if file.endswith(".jpg"):
            img = cv2.imread(f"dataset/{file}", cv2.IMREAD_GRAYSCALE)
            id = int(file.split('.')[1])
            faces.append(img)
            ids.append(id)

    if faces:
        recognizer.train(faces, np.array(ids))
        recognizer.save("trainer.yml")
        print("[✓] Model trained successfully.")
        speak("Model training complete")
    else:
        print("[!] No face data found. Register employees first.")
        speak("No face data found. Please register employees first")

def start_attendance():
    print("\n[STARTING ATTENDANCE]")
    speak("Starting attendance system")
    if not os.path.exists("trainer.yml"):
        print("[!] No trained model found. Please train first.")
        speak("No trained model found. Please train the model first")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    students = load_students()
    marked = set()

    cap = cv2.VideoCapture(0)
    print("[INFO] Camera started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture video.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 60 and id in students:
                name = students[id]
                if name not in marked:
                    save_attendance(name)
                    marked.add(name)
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[✓] Attendance session ended.")
    speak("Attendance session ended")

# ========== MENU ==========
def menu():
    speak("Welcome to the Face Attendance System")
    while True:
        print_banner()
        print("1. Register New Employee")
        print("2. Train Model")
        print("3. Start Attendance")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            register_face()
        elif choice == '2':
            train_model()
        elif choice == '3':
            start_attendance()
        elif choice == '4':
            speak("Goodbye!")
            print("[EXIT] Goodbye!")
            break
        else:
            print("[!] Invalid choice. Please try again.")
            speak("Invalid choice, please try again")

# ========== RUN ==========
if _name_ == "_main_":
    menu()