import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Setting up the layout with columns
col1, col2, col3 = st.columns([1,6,1])
with col1:
    st.write("")

# with col2:
#     st.image(".\Images\logo.png" , width=530, use_column_width=True)

with col3:
    st.write("")

# Title and introduction
st.title("Musicify")
st.write("This web application is an emotion-based music recommendation system. To get recommended songs, start by allowing mic and camera access for this web application.")

# Load pre-trained models and data
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
hol = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Check and initialize session state variables
if "run" not in st.session_state:
    st.session_state["run"] = "true"
try:
    detected_emotion = np.load("detected_emotion.npy")[0]
except:
    detected_emotion = ""

if not(detected_emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"

# Class for Emotion Detector
class EmotionDetector:
    def recv(self, frame):    
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = hol.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)) 
        lst = []

        # Extracting facial and hand landmarks
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
                for i in range(42):
                    lst.append(0.0)

        if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
                for i in range(42):
                    lst.append(0.0)
        
        lst = np.array(lst).reshape(1,-1)

        # Predict emotion using the loaded model
        pred = label[np.argmax(model.predict(lst))]
        print(pred)

        # Display the detected emotion on the video frame
        cv2.putText(frm, pred, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
        np.save("detected_emotion.npy", np.array([pred])) 

        # Draw landmarks on the video frame
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# User input for language and artist preferences
lang = st.text_input("Enter Your Preferred Language")
artist = st.text_input("Enter Your Preferred Artist")

# Display the webcam feed and emotion detection results if preferences are provided
# if lang and artist and st.session_state["run"] != "false":
webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionDetector)

# Button to recommend music based on detected emotion, language, and artist preferences
btn = st.button("Recommend Music")

if btn:
	if not(detected_emotion):
		st.warning("Please Let Me Capture Your Emotion First")
		st.session_state["run"] = "true"
	else:
		webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{detected_emotion}+songs+{artist}")
		np.save("detected_emotion.npy", np.array([""]))
		st.session_state["run"] = "false"

# Display credits
st.write('Made with ❤ by Mrinalika, Nishant, Rishabh, Mitali, Devesh, Santoshi')

# Streamlit customization to hide header and footer
st.markdown(
      """
      <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
      </style>
      """,
      unsafe_allow_html=True
)