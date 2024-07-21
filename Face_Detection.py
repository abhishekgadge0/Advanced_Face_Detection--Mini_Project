import cv2
import os
import shutil
import numpy as np

# Step 1: Initialization
# Directories for storing captured face images and trained models
faces_dir = 'captured_faces'
if not os.path.exists(faces_dir):
    os.makedirs(faces_dir)

model_dir = 'trained_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load age and gender models
age_model_dir = '/Users/abhi./Desktop/My Computer/Projects✅/Face_detection/Req. Models'  # Update with the correct path to the directory
gender_model_dir = '/Users/abhi./Desktop/My Computer/Projects✅/Face_detection/Req. Models'  # Update with the correct path to the directory

deploy_age_prototxt = os.path.join(age_model_dir, 'age_deploy.prototxt')
age_caffemodel = os.path.join(age_model_dir, 'age_net.caffemodel')
deploy_gender_prototxt = os.path.join(gender_model_dir, 'gender_deploy.prototxt')
gender_caffemodel = os.path.join(gender_model_dir, 'gender_net.caffemodel')

# Check if the model files exist
if not (os.path.exists(deploy_age_prototxt) and os.path.exists(age_caffemodel)):
    raise FileNotFoundError(f"Age model files not found in {age_model_dir}")

if not (os.path.exists(deploy_gender_prototxt) and os.path.exists(gender_caffemodel)):
    raise FileNotFoundError(f"Gender model files not found in {gender_model_dir}")

# Load the models
age_net = cv2.dnn.readNetFromCaffe(deploy_age_prototxt, age_caffemodel)
gender_net = cv2.dnn.readNetFromCaffe(deploy_gender_prototxt, gender_caffemodel)

# Mean values for model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Age and gender list
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Step 2: Face Detection and Image Capture
def capture_faces(person_name):
    person_dir = os.path.join(faces_dir, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    video_cap = cv2.VideoCapture(0)
    face_count = 0

    while True:
        ret, frame = video_cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_count += 1
            face_img = gray[y:y+h, x:x+w]
            face_path = os.path.join(person_dir, f'{person_name}_{face_count}.jpg')
            cv2.imwrite(face_path, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Capture Faces', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or face_count >= 100:
            break

    video_cap.release()
    cv2.destroyAllWindows()
    print(f"Total faces captured for {person_name}: {face_count}")

# Step 3: Train a Face Recognizer
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}
    current_label = 0
    
    for person_name in os.listdir(faces_dir):
        person_dir = os.path.join(faces_dir, person_name)
        if os.path.isdir(person_dir):
            for face_filename in os.listdir(person_dir):
                if face_filename.endswith('.jpg'):
                    face_path = os.path.join(person_dir, face_filename)
                    face_img = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
                    faces.append(face_img)
                    labels.append(current_label)
            label_map[current_label] = person_name
            current_label += 1
    
    recognizer.train(faces, np.array(labels))
    recognizer.save(os.path.join(model_dir, 'face_recognizer.yml'))
    np.save(os.path.join(model_dir, 'label_map.npy'), label_map)
    print("Face recognizer trained and model saved.")

# Step 4: Face Recognition with Age and Gender Prediction
def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(model_dir, 'face_recognizer.yml'))
    label_map = np.load(os.path.join(model_dir, 'label_map.npy'), allow_pickle=True).item()
    
    video_cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video_cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_img)
            if confidence < 100:  # Adjust confidence threshold as needed
                person_name = label_map[label]

                # Predict Age and Gender
                face_blob = cv2.dnn.blobFromImage(frame[y:y+h, x:x+w], 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                
                gender_net.setInput(face_blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                
                age_net.setInput(face_blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                
                label_text = f'{person_name}, {gender}, {age}'
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow('Recognize Faces', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_cap.release()
    cv2.destroyAllWindows()

# Step 5: Delete Captured Data
def delete_data():
    print("Options:")
    print("1: Delete all captured data")
    print("2: Delete data for a specific person")
    choice = input("Enter your choice: ")
    
    if choice == '1':
        shutil.rmtree(faces_dir)
        shutil.rmtree(model_dir)
        os.makedirs(faces_dir)
        os.makedirs(model_dir)
        print("All captured data has been deleted.")
    elif choice == '2':
        person_name = input("Enter the person's name: ")
        person_dir = os.path.join(faces_dir, person_name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
            print(f"Data for {person_name} has been deleted.")
        else:
            print(f"No data found for {person_name}.")
    else:
        print("Invalid choice.")

# Main Function to Run the Project
def main():
    try:
        while True:
            print("Options:")
            print("1: Capture faces")
            print("2: Train face recognizer")
            print("3: Recognize faces")
            print("4: Delete captured data")
            print("5: Exit")
            
            choice = input("Enter your choice: ")
            
            if choice == '1':
                person_name = input("Enter the person's name: ")
                print(f"Capturing faces for {person_name}...")
                capture_faces(person_name)
            
            elif choice == '2':
                print("Training face recognizer...")
                train_recognizer()
            
            elif choice == '3':
                print("Recognizing faces...")
                recognize_faces()
            
            elif choice == '4':
                print("Deleting captured data...")
                delete_data()
            
            elif choice == '5':
                break
            
            else:
                print("Invalid choice. Please try again.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        print("All OpenCV windows closed.")

if __name__ == "__main__":
    main()
