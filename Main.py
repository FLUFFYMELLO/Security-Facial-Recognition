import cv2
import face_recognition
import json
import streamlit as st

# Load profiles JSON
with open("profiles/student_profiles.json") as f:
    profiles = json.load(f)

# Precompute encodings
known_encodings = []
known_profiles = []
for profile in profiles:
    img = face_recognition.load_image_file(profile["image"])
    encoding = face_recognition.face_encodings(img)[0]
    known_encodings.append(encoding)
    known_profiles.append(profile)

Camera = cv2.VideoCapture(0)

st.title("Student Face Recognition")

while True:
    ret, frame = Camera.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    st.image(rgb_frame, channels="RGB")

    for encoding in face_encodings:
        results = face_recognition.compare_faces(known_encodings, encoding)
        if True in results:
            idx = results.index(True)
            profile = known_profiles[idx]

            # Show profile card
            st.image(profile["image"], caption=profile["name"])
            st.write(f"**ID:** {profile['id']}")
            st.write(f"**Role:** {profile['role']}")
            st.success(f"{profile['name']} recognized")
        else:
            st.error("Not a student")

    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

Camera.release()
cv2.destroyAllWindows()