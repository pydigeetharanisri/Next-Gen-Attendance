import streamlit as st
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import face_recognition
import pickle
import pandas as pd
import time
from PIL import Image
import base64

# Set page config
st.set_page_config(
    page_title="NEXT-GEN ATTENDANCE",
    layout="wide"
)

def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("Background image not found. Using default background.")
    except Exception as e:
        st.warning(f"Could not load background image: {str(e)}")

class TimeTable:
    def __init__(self):
        # Define time periods
        self.periods = {
            1: ("09:30", "10:30"),
            2: ("10:30", "11:20"),
            3: ("11:20", "12:10"),
            4: ("12:10", "13:00"),
            5: ("14:00", "14:50"),
            6: ("14:50", "15:40"),
            7: ("15:40", "16:30")
        }
        
        # Define subject faculty mapping
        self.subject_faculty = {
            "OS": "A LEELAVATHI",
            "DM": "G PRASANTHI",
            "WT": "L ATRIDATTA RAVITEZ",
            "MCCP-I": "Dr. V.VENKATESWARA RAO,KADALI RAMYA",
            "STM": "Faculty Name",
            "PPL": "Faculty Name",
            "AI": "S KUMAR REDDY MALLIDI",
            "CG": "Faculty Name",
            "PCS-III": "J N V SOMAYAJULU,AMARLAPUDI KIRANMAYEE",
            "DM LAB": "G PRASANTHI,G SRI RAM GANESH",
            "WT LAB": "L ATRIDATTA RAVITEZ,YENTRAPATI SABITHA YALI",
            "SOC-III": "Faculty Name",
            "INTERNSHIP": "Faculty Name",
            "LIBRARY": "Dr. G CH S MADHUSUDHAN RAO",
            "SPORTS": "Faculty Name",
            "CSP": "Faculty Name"
        }
        
        # Define timetable
        self.timetable = {
            'Monday': {1: "PCS-III", 2: "PCS-III", 3: "AI", 4: "DM", 5: "", 6: "MCCP-I", 7: "MCCP-I"},
            'Tuesday': {1: "OS", 2: "DM LAB", 3: "DM LAB", 4: "DM LAB", 5: "", 6: "WT", 7: "DM"},
            'Wednesday': {1: "DM", 2: "WT", 3: "WT", 4: "LIBRARY", 5: "", 6: "OS", 7: "SPORTS"},
            'Thursday': {1: "AI", 2: "OS", 3: "DM", 4: "OS", 5: "", 6: "WT LAB", 7: "WT LAB"},
            'Friday': {1: "OS", 2: "DM", 3: "WT", 4: "AI", 5: "", 6: "MCCP-I", 7: "MCCP-I"},
            'Saturday': {1: "WT", 2: "DM", 3: "PCS-III", 4: "PCS-III", 5: "", 6: "OS", 7: "WT"}
        }

    def get_current_period(self):
        current_time = datetime.now().time()
        for period, (start_str, end_str) in self.periods.items():
            start_time = datetime.strptime(start_str, "%H:%M").time()
            end_time = datetime.strptime(end_str, "%H:%M").time()
            if start_time <= current_time <= end_time:
                return period
        return None

    def get_current_subject(self):
        current_day = datetime.now().strftime("%A")
        current_period = self.get_current_period()
        
        if current_period is None:
            return None
            
        if current_day in self.timetable and current_period in self.timetable[current_day]:
            subject = self.timetable[current_day][current_period]
            return subject if subject != "" else None
        return None

    def get_subject_faculty(self, subject):
        return self.subject_faculty.get(subject, "Unknown Faculty")

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize essential paths and create directories
        self.dataset_path = "dataset"
        self.model_path = "trained_model.pkl"
        self.attendance_path = "attendance"
        self.timetable = TimeTable()
        
        # Create necessary directories if they don't exist
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.attendance_path, exist_ok=True)

    def check_camera(self):
        """Test camera availability"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return False
            cap.release()
            return True
        except Exception:
            return False

    def create_dataset(self, person_name, status_text, progress_bar, frame_placeholder):
        cap = None
        try:
            if not self.check_camera():
                raise Exception("Camera not available!")
            
            person_dir = os.path.join(self.dataset_path, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            count = 0
            total_images = 30
            
            status_text.text("Please look at the camera and move your face slightly...")
            
            while count < total_images:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to grab frame from camera")
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    if len(faces) == 1:  # Only save if one face is detected
                        count += 1
                        face_img = frame[y:y+h, x:x+w]
                        file_name = os.path.join(person_dir, f"{person_name}_{count}.jpg")
                        cv2.imwrite(file_name, face_img)
                        
                        # Update progress
                        progress = count / total_images
                        progress_bar.progress(progress)
                        status_text.text(f"Capturing image {count}/{total_images}")
                        time.sleep(0.2)
                
                # Convert BGR to RGB for displaying in Streamlit
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            return True
            
        except Exception as e:
            status_text.error(f"Error: {str(e)}")
            return False
        finally:
            if cap is not None:
                cap.release()

    def train_model(self, progress_text, progress_bar):
        try:
            known_faces = []
            known_names = []
            
            if not os.path.exists(self.dataset_path):
                raise Exception("Dataset folder not found! Please create dataset first.")
            
            person_dirs = [d for d in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, d))]
            
            total_persons = len(person_dirs)
            if total_persons == 0:
                raise Exception("No datasets found! Please create dataset first.")
                
            current_person = 0
            
            for person_name in person_dirs:
                current_person += 1
                progress = current_person / total_persons
                progress_bar.progress(progress)
                progress_text.text(f"Processing {person_name}...")
                
                person_dir = os.path.join(self.dataset_path, person_name)
                person_encodings = []
                
                image_files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                for image_name in image_files:
                    image_path = os.path.join(person_dir, image_name)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image)
                        if face_encodings:
                            person_encodings.append(face_encodings[0])
                    except Exception as e:
                        st.warning(f"Error processing {image_path}: {str(e)}")
                        continue
                
                if person_encodings:
                    # Use average encoding for better accuracy
                    average_encoding = np.mean(person_encodings, axis=0)
                    known_faces.append(average_encoding)
                    known_names.append(person_name)
                else:
                    st.warning(f"No valid face encodings found for {person_name}")
            
            if not known_faces:
                raise Exception("No face encodings could be generated from the dataset!")
                
            model_data = {
                "faces": known_faces,
                "names": known_names
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            progress_text.text(f"Training completed! Trained on {len(known_names)} students.")
            return True
            
        except Exception as e:
            progress_text.error(f"Error: {str(e)}")
            return False

    def mark_attendance(self, subject_name, status_text, frame_placeholder):
        cap = None
        try:
            if not self.check_camera():
                raise Exception("Camera not available!")
            
            if not os.path.exists(self.model_path):
                raise Exception("Model not trained! Please train the model first.")
            
            if subject_name is None or subject_name.strip() == "":
                current_subject = self.timetable.get_current_subject()
                if current_subject is None:
                    raise Exception("No scheduled class at current time! Please specify a subject.")
                subject_name = current_subject
            
            faculty_name = self.timetable.get_subject_faculty(subject_name)
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            known_faces = model_data["faces"]
            known_names = model_data["names"]
            
            cap = cv2.VideoCapture(0)
            
            date = datetime.now().strftime("%Y-%m-%d")
            date_dir = os.path.join(self.attendance_path, date)
            os.makedirs(date_dir, exist_ok=True)
            
            attendance_file = os.path.join(date_dir, f"{subject_name}_attendance.csv")
            all_records = self.get_attendance_records(date)
            
            subject_attendance = set()
            if not all_records.empty:
                subject_records = all_records[all_records['Subject'] == subject_name]
                if not subject_records.empty:
                    subject_attendance = set(subject_records['Name'].unique())
            
            start_time = datetime.now()
            time_limit = timedelta(seconds=10)
            
            status_text.info(f"Starting attendance for {subject_name} (Faculty: {faculty_name})")
            
            while (datetime.now() - start_time) < time_limit:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to grab frame from camera")
                
                # Resize frame for faster face recognition
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_names[first_match_index]
                        
                        if name not in subject_attendance:
                            time_now = datetime.now().strftime("%H:%M:%S")
                            
                            df = pd.DataFrame([[name, date, time_now, subject_name, faculty_name]], 
                                           columns=['Name', 'Date', 'Time', 'Subject', 'Faculty'])
                            
                            if os.path.exists(attendance_file):
                                df.to_csv(attendance_file, mode='a', header=False, index=False)
                            else:
                                df.to_csv(attendance_file, index=False)
                            
                            subject_attendance.add(name)
                            status_text.success(f"✓ Attendance marked for {name} in {subject_name}")
                        else:
                            status_text.warning(f"! {name} already marked for {subject_name} today")
                        
                        # Draw rectangle and name (scaled back to original size)
                        top, right, bottom, left = face_location
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, top - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                
                remaining_time = 10 - int((datetime.now() - start_time).total_seconds())
                if remaining_time > 0:
                    status_text.text(f"Time remaining: {remaining_time}s")
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB", use_column_width=True)
            
            return True
            
        except Exception as e:
            status_text.error(f"Error: {str(e)}")
            return False
        finally:
            if cap is not None:
                cap.release()

    def get_attendance_records(self, date):
        try:
            date_dir = os.path.join(self.attendance_path, date)
            all_records = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Subject', 'Faculty'])
            
            if os.path.exists(date_dir):
                csv_files = [f for f in os.listdir(date_dir) if f.endswith('_attendance.csv')]
                
                for file in csv_files:
                    file_path = os.path.join(date_dir, file)
                    try:
                        df = pd.read_csv(file_path)
                        if not df.empty:
                            all_records = pd.concat([all_records, df], ignore_index=True)
                    except Exception as e:
                        st.warning(f"Error reading {file}: {str(e)}")
            
            return all_records
        except Exception as e:
            st.error(f"Error getting attendance records: {str(e)}")
            return pd.DataFrame(columns=['Name', 'Date', 'Time', 'Subject', 'Faculty'])

def main():
    st.title("NEXT-GEN ATTENDANCE 📸")
    
    # Try to add background image
    add_bg_from_local('assets/hm2.jpg')
    
    # Initialize system
    system = FaceRecognitionSystem()
    
    st.markdown("""
        <style>
        .main-title {
            color: #FFFFFF;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            text-shadow: 2px 2px 4px #000000;
            padding: 20px;
        }
        .tagline {
            color: #FFFFFF;
            font-size: 24px;
            text-align: center;
            font-style: italic;
            text-shadow: 1px 1px 2px #000000;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: rgba(255, 255, 255, 0.1);
            border: 2px solid white;
        }
        .stSelectbox, .stTextInput {
            background-color: rgba(255, 255, 255, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Register", "🎯 Train Model", "✅ Mark Attendance", "📊 View Records"])
    
    # Register New Student Tab
    with tab1:
        st.header("Register New Student")
        person_name = st.text_input("Enter student name (e.g., 20B01A0501)")
        
        if st.button("Create Dataset"):
            if person_name and person_name.strip():
                status_text = st.empty()
                progress_bar = st.progress(0)
                frame_placeholder = st.empty()
                
                if system.create_dataset(person_name.strip(), status_text, progress_bar, frame_placeholder):
                    st.success(f"Dataset created successfully for {person_name}")
                else:
                    st.error("Failed to create dataset")
            else:
                st.warning("Please enter a valid name")
    
    # Train Model Tab
    with tab2:
        st.header("Train Recognition Model")
        if st.button("Train Model"):
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            if system.train_model(progress_text, progress_bar):
                st.success("Model trained successfully!")
            else:
                st.error("Model training failed!")
    
    # Mark Attendance Tab
    with tab3:
        st.header("Mark Attendance")
        
        # Get current subject from timetable
        current_subject = system.timetable.get_current_subject()
        if current_subject:
            st.info(f"Current scheduled subject: {current_subject}")
        else:
            st.info("No class scheduled at current time")
        
        # Available subjects for dropdown
        available_subjects = list(system.timetable.subject_faculty.keys())
        
        # Option to select or override current subject
        col1, col2 = st.columns(2)
        
        with col1:
            subject_dropdown = st.selectbox(
                "Select Subject:", 
                [""] + available_subjects,
                index=0 if not current_subject else available_subjects.index(current_subject) + 1 if current_subject in available_subjects else 0
            )
        
        with col2:
            subject_override = st.text_input("Or enter custom subject code:")
        
        if st.button("Start Attendance"):
            status_text = st.empty()
            frame_placeholder = st.empty()
            
            # Priority: custom input > dropdown > current scheduled
            subject_to_use = None
            if subject_override and subject_override.strip():
                subject_to_use = subject_override.strip()
            elif subject_dropdown:
                subject_to_use = subject_dropdown
            elif current_subject:
                subject_to_use = current_subject
            
            if subject_to_use:
                if system.mark_attendance(subject_to_use, status_text, frame_placeholder):
                    st.success("Attendance marking completed!")
                else:
                    st.error("Failed to mark attendance")
            else:
                st.error("Please select or enter a subject code before starting attendance!")
    
    # View Records Tab
    with tab4:
        st.header("View Attendance Records")
        
        # Date selector
        selected_date = st.date_input("Select date", datetime.now())
        date_str = selected_date.strftime("%Y-%m-%d")
        
        # Get records for selected date
        records = system.get_attendance_records(date_str)
        
        if not records.empty:
            # Show summary statistics
            st.subheader("Summary Statistics")
            total_students = len(records['Name'].unique())
            total_subjects = len(records['Subject'].unique())
            st.write(f"Total students marked: {total_students}")
            st.write(f"Total subjects: {total_subjects}")
            
            # Show detailed records
            st.subheader("Detailed Records")
            
            # Group by subject
            subjects = records['Subject'].unique()
            for subject in subjects:
                subject_records = records[records['Subject'] == subject]
                
                # Check if subject_records is not empty before accessing iloc[0]
                if not subject_records.empty:
                    st.write(f"\n**Subject:** {subject}")
                    faculty_name = subject_records['Faculty'].iloc[0] if len(subject_records) > 0 else "Unknown"
                    st.write(f"**Faculty:** {faculty_name}")
                    st.write(f"**Students present:** {len(subject_records)}")
                    
                    # Display student details in a table
                    subject_records_display = subject_records[['Name', 'Time']].rename(
                        columns={'Name': 'Roll Number', 'Time': 'Marked At'}
                    )
                    st.dataframe(subject_records_display, use_container_width=True)
                    
                    # Option to download subject-wise attendance
                    csv = subject_records.to_csv(index=False)
                    st.download_button(
                        label=f"Download {subject} attendance",
                        data=csv,
                        file_name=f"{date_str}_{subject}_attendance.csv",
                        mime="text/csv",
                        key=f"download_{subject}"
                    )
                    
                    st.divider()  # Add separator between subjects
        else:
            st.info("No attendance records found for selected date")

if __name__ == "__main__":
    main()