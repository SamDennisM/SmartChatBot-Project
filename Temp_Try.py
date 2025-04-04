import sys
import torch
import cv2
import ollama
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from yolov5 import YOLOv5

class AudioRecorderThread(QThread):
    text_update = pyqtSignal(str)
    finished_recording = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        self.recognizer = sr.Recognizer()
        self.audio_buffer = []
        self.running = False

    def run(self):
        self.running = True
        self.audio_buffer.clear()

        print("Listening...")
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
            callback=self.audio_callback
        )

        with stream:
            while self.running:
                sd.sleep(500)

        if self.audio_buffer:
            self.process_transcription()

    def audio_callback(self, indata, frames, time, status):
        if self.running:
            self.audio_buffer.append(indata.copy())

    def process_transcription(self):
        """Convert recorded audio into text and send to chatbot."""
        audio_data = np.concatenate(self.audio_buffer, axis=0)
        audio_data = np.array(audio_data, dtype=np.int16)
        audio = sr.AudioData(audio_data.tobytes(), self.sample_rate, 2)

        try:
            print("Transcribing...")
            text = self.recognizer.recognize_google(audio)
            self.text_update.emit(text)
            self.finished_recording.emit(text)  # Send text to chatbot
        except sr.UnknownValueError:
            self.text_update.emit("Could not understand the speech.")
            self.finished_recording.emit("")
        except sr.RequestError:
            self.text_update.emit("Speech Recognition service error.")
            self.finished_recording.emit("")

    def stop(self):
        self.running = False
        self.wait()

class VideoCaptureThread(QThread):
    frame_captured = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0)
        self.yolo_model = YOLOv5('yolov5s.pt', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                results = self.yolo_model.predict(frame)
                frame_with_boxes = results.render()[0]

                height, width, _ = frame_with_boxes.shape
                q_image = QImage(frame_with_boxes.data, width, height, 3 * width, QImage.Format_BGR888)
                self.frame_captured.emit(q_image)

    def stop(self):
        self.running = False
        self.capture.release()
        self.wait()

class SmartClassroomAssistant(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.audio_thread = None  
        self.video_thread = VideoCaptureThread()
        self.video_thread.frame_captured.connect(self.update_frame)
        self.video_thread.start()
        self.chat_history = []  

    def initUI(self):
        self.setWindowTitle('Smart Classroom Assistant - AI Chatbot & Real-Time Detection')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.video_label = QLabel(self)
        layout.addWidget(self.video_label)

        self.transcription_display = QTextEdit(self)
        self.transcription_display.setReadOnly(True)
        self.transcription_display.setPlaceholderText("Transcription will appear here...")
        layout.addWidget(self.transcription_display)

        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Assistant responses will appear here...")
        layout.addWidget(self.chat_display)

        self.push_to_talk_btn = QPushButton('🎤 Start Listening', self)
        self.push_to_talk_btn.clicked.connect(self.start_recording)
        layout.addWidget(self.push_to_talk_btn)

        self.stop_listening_btn = QPushButton('🛑 Stop Listening', self)
        self.stop_listening_btn.clicked.connect(self.stop_recording)
        self.stop_listening_btn.setEnabled(False)
        layout.addWidget(self.stop_listening_btn)

        self.stop_btn = QPushButton('Exit', self)
        self.stop_btn.clicked.connect(self.stop_application)
        layout.addWidget(self.stop_btn)

        self.setLayout(layout)

    def update_frame(self, q_image):
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def update_transcription(self, text):
        """Update the transcription display."""
        self.transcription_display.setPlainText(f"Live Transcription:\n{text}")

    def send_to_chatbot(self, text):
        """Send transcribed text to chatbot and display the response."""
        if text.strip():
            self.chat_display.append(f"You: {text}")
            bot_response = self.chat_with_ollama(text)
            self.chat_display.append(f"Assistant: {bot_response}")

    def start_recording(self):
        """Start real-time speech-to-text."""
        if self.audio_thread is not None:
            self.audio_thread.stop()

        self.audio_thread = AudioRecorderThread()
        self.audio_thread.text_update.connect(self.update_transcription)
        self.audio_thread.finished_recording.connect(self.send_to_chatbot)  # Send transcript to chatbot

        self.transcription_display.clear()
        self.transcription_display.setPlaceholderText("Listening... Speak now")
        self.push_to_talk_btn.setEnabled(False)
        self.stop_listening_btn.setEnabled(True)
        self.audio_thread.start()

    def stop_recording(self):
        """Stop recording and process the final transcription."""
        if self.audio_thread:
            self.audio_thread.stop()
        self.push_to_talk_btn.setEnabled(True)
        self.stop_listening_btn.setEnabled(False)
        self.transcription_display.setPlaceholderText("Transcription will appear here...")

    def chat_with_ollama(self, user_input, model="mistral"):
        """Send user input to the Ollama chatbot and get a response."""
        self.chat_history.append({"role": "user", "content": user_input})

        try:
            response = ollama.chat(model=model, messages=self.chat_history)
            bot_reply = response.get('message', {}).get('content', "[No response received]")
            self.chat_history.append({"role": "assistant", "content": bot_reply})
            return bot_reply
        except ollama._types.ResponseError as e:
            return f"Ollama Error: {e}"
        except Exception as e:
            return f"Unexpected Error: {e}"

    def stop_application(self):
        """Clean up and exit."""
        if self.audio_thread:
            self.audio_thread.stop()
        self.video_thread.stop()
        self.close()

    def closeEvent(self, event):
        """Cleanup when the window is closed."""
        if self.audio_thread:
            self.audio_thread.stop()
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmartClassroomAssistant()
    window.show()
    sys.exit(app.exec_())
