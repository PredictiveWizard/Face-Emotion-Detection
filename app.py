from flask import Flask, render_template, Response
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Flask app
app = Flask(__name__)

# Load model and classifier
face_classifier = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
classifier = load_model(r"model.h5")

emotion_labels = ['Furious','Incurious','Nervous','Joyful','Neutral','Sad','Exclaimed!']

# Camera capture
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    cv2.putText(frame, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Stop camera gracefully on shutdown
@app.route('/stop')
def stop():
    cap.release()
    cv2.destroyAllWindows()
    return "Camera stopped."

if __name__ == "__main__":
    app.run(debug=True)
