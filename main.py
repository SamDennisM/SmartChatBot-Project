import sys
import torch
import cv2
import ollama
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import streamlit as st
from yolov5 import YOLOv5

class SmartClassroomAssistant:
    def __init__(self):
        self.yolo_model = YOLOv5('yolov5s.pt', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.recognizer = sr.Recognizer()
        self.sample_rate = 16000
        self.audio_buffer = []
        self.chat_history = []
        self.video_capture = cv2.VideoCapture(0)
        self.listening = False

    def process_transcription(self, audio_data):
        try:
            text = self.recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the speech."
        except sr.RequestError:
            return "Speech Recognition service error."

    def chat_with_ollama(self, user_input, model="mistral"):
        self.chat_history.append({"role": "user", "content": user_input})
        try:
            response = ollama.chat(model=model, messages=self.chat_history)
            bot_reply = response.get('message', {}).get('content', "[No response received]")
            self.chat_history.append({"role": "assistant", "content": bot_reply})
            return bot_reply
        except Exception as e:
            return f"Error: {e}"

    def detect_objects(self, frame):
        results = self.yolo_model.predict(frame)
        return results.render()[0]

assistant = SmartClassroomAssistant()
st.set_page_config(page_title="Smart Classroom Assistant", layout="wide")
st.title("📚 Smart Classroom Assistant 🎤")

# Layout
col1, col2 = st.columns([3, 2])

# Video Capture in the left column
with col1:
    st.subheader("📷 Live Camera Feed")
    ret, frame = assistant.video_capture.read()
    if ret:
        processed_frame = assistant.detect_objects(frame)
        st.image(processed_frame, channels="BGR", use_container_width=True)
    
    if st.button("🛑 Stop Camera"):
        assistant.video_capture.release()
        st.stop()

# Chatbot section in the right column
with col2:
    st.subheader("💬 Chatbot Interaction")
    chat_container = st.container()
    transcription_container = st.container()

    for chat in assistant.chat_history:
        role = "🧑 You:" if chat["role"] == "user" else "🤖 Assistant:"
        chat_container.write(f"{role} {chat['content']}")

    if st.button("🎤 Start Listening"):
        assistant.listening = True
        with sd.InputStream(samplerate=assistant.sample_rate, channels=1, dtype=np.int16) as stream:
            audio_data = np.concatenate([stream.read(500)[0] for _ in range(10)], axis=0)
            audio = sr.AudioData(audio_data.tobytes(), assistant.sample_rate, 2)
            transcript = assistant.process_transcription(audio)
            
            if transcript.strip():
                transcription_container.write(f"📝 Transcribed Text: {transcript}")
                assistant.chat_history.append({"role": "user", "content": transcript})
                response = assistant.chat_with_ollama(transcript)
                assistant.chat_history.append({"role": "assistant", "content": response})
                
                st.rerun()
    
    if st.button("🛑 Stop Listening"):
        assistant.listening = False
