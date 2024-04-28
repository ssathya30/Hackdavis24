import tkinter as tk
import customtkinter as ck
import pandas as pd
import numpy as np
import joblib
import mediapipe as mp
import os
import cv2
from PIL import Image, ImageTk
import sounddevice as sd
import soundfile as sf


from landmarks import landmarks
import yagmail
import os
import threading

from twilio.rest import Client
import simpleaudio as sa

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
fs = 40100  # Sample rate (adjust as needed)
chunk_size = int(fs / 10)  # Chunk size (0.1 seconds)
max_duration = 5.0  # Maximum duration in seconds
audio_buffer = []  # Buffer to store the audio samples
warningCounter = 0
Counter = 0
counter = 0
contents = [
    "This is a message informing you that Bob is suffering a potential medical emergency. Please try to contact him."
]

account_sid = 'sid'
auth_token = 'token'
client = Client(account_sid, auth_token)

def record_audio():
    global audio_buffer
    recorded_duration = 0.0

    # Open the default microphone for recording
    audio_data = sd.rec(int(max_duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to complete

    if len(audio_buffer) > 0:
        audio_buffer.append(audio_data)
    else:
        audio_buffer = [audio_data]

    audio_np = np.concatenate(audio_buffer, axis=0)
    variance = np.var(audio_np)
    print("Variance:", variance)

    # Clear the audio buffer to prepare for the next recording
    audio_buffer = []

    # Start a new recording if needed
    if recorded_duration < max_duration:
        threading.Timer(chunk_size / fs, record_audio).start()

def playSound():
    try:
        # Load the WAV file
        wave_obj = sa.WaveObject.from_wave_file('./alarm.wav')

        play_thread = threading.Thread(target=wave_obj.play)
        play_thread.start()

    except Exception as e:
        print(f"Error playing WAV file: {e}")

yag = yagmail.SMTP("nabouali@profilesag.com")
window = tk.Tk()
window.geometry("1000x1000")
window.title("Position Detector")
ck.set_appearance_mode("dark")

# Create GUI labels and buttons using grid layout
classLabel = ck.CTkLabel(window, text="Status", font=("Arial", 20), text_color="black")
classLabel.grid(row=0, column=0, padx=10, pady=10)

probLabel = ck.CTkLabel(window, text="PROB", font=("Arial", 20), text_color="black")
probLabel.grid(row=0, column=2, padx=10, pady=10)

classBox = ck.CTkLabel(window, text="0", font=("Arial", 20), text_color="black")
classBox.grid(row=1, column=0, padx=10, pady=10)

counterBox = ck.CTkLabel(window, text="0", font=("Arial", 20), text_color="black")
counterBox.grid(row=1, column=1, padx=10, pady=10)

probBox = ck.CTkLabel(window, text="0", font=("Arial", 20), text_color="black")
probBox.grid(row=1, column=2, padx=10, pady=10)

reset_button = ck.CTkButton(window, command=lambda: counterBox.configure(text='0'), text="Reset", font=("Arial", 20), text_color="black")
reset_button.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

cap = cv2.VideoCapture(0)  
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def detect():
    global counter
    global warningCounter

    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    

    if results.pose_landmarks:
        # Draw landmark on body
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        
        right_shoulder = None
        left_shoulder = None
        righteyeAvg = [0.0, 0.0, 0.0, 0.0]
        lefteyeAvg = [0.0, 0.0, 0.0, 0.0]
        # record_audio()
        j = 0
        k = 0
        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks)
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                if  i == 1 or i == 2 or i == 3:
                    righteyeAvg = [landmark.x + righteyeAvg[0], landmark.y + righteyeAvg[1], landmark.z + righteyeAvg[2], landmark.visibility]
                    j += 1
                elif  i == 4 or i == 5 or i == 6:
                    righteyeAvg = [landmark.x + lefteyeAvg[0], landmark.y + lefteyeAvg[1], landmark.z + lefteyeAvg[2], landmark.visibility] 
                    k += 1
                elif i == 11:  
                    left_shoulder = [landmark.x, landmark.y, landmark.z, landmark.visibility]
                elif i == 12:  
                    right_shoulder = [landmark.x, landmark.y, landmark.z, landmark.visibility]

            if (j > 0 and k > 0):
                # do smtg
                righteyeAvg[0] = righteyeAvg[0] / j
                righteyeAvg[1] = righteyeAvg[1] / j
                righteyeAvg[2] = righteyeAvg[2] / j 

                lefteyeAvg[0] = lefteyeAvg[0] / k
                lefteyeAvg[1] = lefteyeAvg[1] / k
                lefteyeAvg[2] = lefteyeAvg[2] / k


            if left_shoulder is not None and right_shoulder is not None:
                if (abs(left_shoulder[1] - right_shoulder[1]) > 0.15):
                    warningCounter += 1
                else:
                    warningCounter = 0
            
            if warningCounter > 10:
                playSound()
                yag.send('nabouali@profilesag.com', 'subject', contents)
                # message = client.messages \
                # .create(
                #      body=contents,
                #      from_='+18558073740'
                #      to='+15108579074'
                #  )
                # print(message.status)
                print("Message Sent")
                if abs(righteyeAvg - lefteyeAvg) > 0.01 or (righteyeAvg[0] - righteyeAvg[2]) > 0.005 or (lefteyeAvg[0] - lefteyeAvg[2]) > 0.005:
                    print("Potential Face Droop Detected")
                


        except Exception as e:
            print("Error during pose processing:", e)

    
    img = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    # Frequency
    window.after(100, detect)

frame = tk.Frame(window)
frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10)
lmain = tk.Label(frame)
lmain.pack()

detect()

window.mainloop()

cap.release()
cv2.destroyAllWindows()
