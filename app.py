# app.py
from flask import Flask, render_template, request, Response, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load face detector model
print("[INFO] loading face detector model...")
prototxtPath = os.path.join("face_detector", "deploy.prototxt")
weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load face mask detector model
print("[INFO] loading face mask detector model...")
maskNet = load_model("mask_detector.model.h5", compile=False)

# Global variable to control webcam streaming
webcam_active = False
camera = None

def detect_and_predict_mask(frame, faceNet, maskNet):
    # Grab the dimensions of the frame and construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Pass the blob through the network and get face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    # Initialize lists for faces, locations, and predictions
    faces = []
    locs = []
    preds = []
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Extract the face ROI, convert it to RGB, resize and preprocess
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    # Only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    
    return (locs, preds)

def process_image(image_path):
    # Read image from file
    image = cv2.imread(image_path)
    
    if image is None:
        return None, "Could not read image"
    
    # Get face detection results
    (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)
    
    # Loop over the detected face locations and their corresponding predictions
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        
        # Determine the class label and color
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # Include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
        # Display the label and bounding box rectangle on the output frame
        cv2.putText(image, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    
    # Create a unique filename to avoid caching issues
    filename = f"processed_{int(time.time())}.jpg"
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(processed_path, image)
    
    return processed_path, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Create a unique filename to avoid conflicts
        filename = f"upload_{int(time.time())}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        processed_image, error = process_image(filepath)
        
        if error:
            return jsonify({'error': error})
        
        return jsonify({
            'original': filepath,
            'processed': processed_image
        })

# For webcam feed
def generate_frames():
    global camera
    # Initialize the camera only when this function is called
    camera = cv2.VideoCapture(0)
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # Process frame for mask detection
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
                
                # Loop over the detected face locations and their corresponding predictions
                for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred
                    
                    # Determine the class label and color
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    
                    # Include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                    
                    # Display the label and bounding box rectangle
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                # Add a frame counter or timestamp for better visualization
                cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Convert frame to JPEG format for streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        # Release the camera when the generator is done
        if camera:
            camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global webcam_active
    webcam_active = True
    return jsonify({'status': 'webcam started'})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global webcam_active, camera
    webcam_active = False
    # Release the camera if it's open
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'webcam stopped'})

@app.route('/webcam_status')
def webcam_status():
    global webcam_active
    return jsonify({'active': webcam_active})

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)
