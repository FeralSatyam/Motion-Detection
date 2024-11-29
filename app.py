import os
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import time
import pygame

app = Flask(__name__)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

pygame.mixer.init()
pygame.mixer.music.load('sound.mp3')

camera = cv2.VideoCapture(0)
motion_detected = False
sound_played = False
motion_detection_enabled = True  # Controls motion detection
recording_enabled = False  # Controls video recording
recording_folder = "Recording"
os.makedirs(recording_folder, exist_ok=True)  # Ensure directory exists
out = None  # OpenCV VideoWriter object

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

def generate_frames():
    global motion_detected, sound_played, motion_detection_enabled, recording_enabled, out

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)

            if motion_detection_enabled:
                motion_detected = (
                    results.face_landmarks or 
                    results.left_hand_landmarks or 
                    results.right_hand_landmarks
                )

                if motion_detected:
                    if not sound_played:
                        pygame.mixer.music.play(loops=-1, start=0.0)
                        sound_played = True

                    cv2.putText(image, "Motion Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    if sound_played:
                        pygame.mixer.music.stop()
                        sound_played = False
            else:
                if sound_played:
                    pygame.mixer.music.stop()
                    sound_played = False

            
            draw_landmarks(image, results)

            if recording_enabled:
                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    filename = os.path.join(recording_folder, f"recording_{int(time.time())}.avi")
                    out = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                out.write(image)
            else:
                if out is not None:
                    out.release()
                    out = None

            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_motion_detection', methods=['POST'])
def toggle_motion_detection():
    global motion_detection_enabled, sound_played
    motion_detection_enabled = not motion_detection_enabled
    if not motion_detection_enabled:
        if sound_played:
            pygame.mixer.music.stop()
            sound_played = False
    return jsonify({"motion_detection_enabled": motion_detection_enabled})

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    global recording_enabled
    recording_enabled = not recording_enabled
    return jsonify({"recording_enabled": recording_enabled})

if __name__ == "__main__":
    app.run(debug=True)
