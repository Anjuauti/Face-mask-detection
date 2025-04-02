import os
import numpy as np
import argparse
import time
import cv2
from imutils.video import VideoStream
import imutils
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

def detect_and_predict_mask(frame, face_net, mask_net, confidence_threshold=0.5):
    """
    Detect faces in a frame and predict whether each face is wearing a mask
    
    Args:
        frame: Input video frame
        face_net: Face detection model
        mask_net: Mask detection model
        confidence_threshold: Minimum detection confidence
        
    Returns:
        tuple: (face locations, mask predictions)
    """
    # Get frame dimensions
    (h, w) = frame.shape[:2]
    
    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Pass the blob through the face detection network
    face_net.setInput(blob)
    detections = face_net.forward()
    
    # Initialize lists for faces, locations, and predictions
    faces = []
    locs = []
    preds = []
    
    # Loop over detected faces
    for i in range(0, detections.shape[2]):
        # Extract confidence score
        confidence = detections[0, 0, i, 2]
        
        # Filter weak detections
        if confidence > confidence_threshold:
            # Compute bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure box is within frame boundaries
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Extract face ROI
            face = frame[startY:endY, startX:endX]
            
            # Check if face exists (avoid empty ROIs)
            if face.size > 0:
                # Preprocess face for mask detection
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)  # Use EfficientNet preprocessing
                
                # Add face and location to lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))
    
    # Make predictions if faces were detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)
    
    return (locs, preds)

def display_stats(frame, mask_count, no_mask_count, start_time):
    """Display stats on the frame"""
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Calculate stats
    total = mask_count + no_mask_count
    mask_percentage = 0 if total == 0 else (mask_count / total) * 100
    
    # Display stats
    stats_bg = np.zeros((100, frame.shape[1], 3), dtype="uint8")
    
    # Stats text
    text = f"Time: {int(elapsed_time)}s | "
    text += f"With Mask: {mask_count} ({mask_percentage:.1f}%) | "
    text += f"Without Mask: {no_mask_count}"
    
    cv2.putText(stats_bg, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Combine stats with frame
    return np.vstack([stats_bg, frame])

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run face mask detection on video stream")
    parser.add_argument("-f", "--face", type=str, default="face_detector",
                        help="path to face detector model directory")
    parser.add_argument("-m", "--model", type=str, default="mask_detector.keras",
                        help="path to trained face mask detector model")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="minimum probability to filter weak detections")
    parser.add_argument("-o", "--output", type=str,
                        help="path to output video file (optional)")
    parser.add_argument("-i", "--input", type=str,
                        help="path to input video file (optional, use webcam if not specified)")
    parser.add_argument("--display", action="store_true", default=True,
                        help="whether to display output frame to screen")
    parser.add_argument("--no-display", dest="display", action="store_false",
                        help="do not display output frame to screen")
    parser.add_argument("--flip-labels", action="store_true", default=False,
                        help="flip the mask/no-mask labels")
    
    args = vars(parser.parse_args())
    
    # Load face detector model
    print("[INFO] Loading face detector model...")
    prototxt_path = os.path.sep.join([args["face"], "deploy.prototxt"])
    weights_path = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)
    
    # Load mask detector model
    print("[INFO] Loading face mask detector model...")
    mask_net = load_model(args["model"])
    
    # Initialize video stream
    print("[INFO] Starting video stream...")
    
    # Use input video file if specified, otherwise use webcam
    if args.get("input"):
        print(f"[INFO] Reading from input video: {args['input']}")
        vs = cv2.VideoCapture(args["input"])
    else:
        print("[INFO] Reading from webcam...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)  # Allow camera to warm up
    
    # Initialize video writer if output path is provided
    writer = None
    
    # Initialize counters
    start_time = time.time()
    mask_count = 0
    no_mask_count = 0
    frame_count = 0
    
    # Option to flip the mask/no-mask labels if needed
    # Set to True based on your feedback
    flip_labels = True
    
    # Loop over frames
    while True:
        # Read frame
        if args.get("input"):
            grabbed, frame = vs.read()
            if not grabbed:
                break
        else:
            frame = vs.read()
            
        frame_count += 1
        
        # Resize frame for faster processing
        frame = imutils.resize(frame, width=640)
        
        # Detect faces and predict mask presence
        (locs, preds) = detect_and_predict_mask(frame, face_net, mask_net, args["confidence"])
        
        # Reset counts for this frame
        frame_mask_count = 0
        frame_no_mask_count = 0
        
        # Process each detection
        for (box, pred) in zip(locs, preds):
            # Unpack bounding box and prediction
            (startX, startY, endX, endY) = box
            
            # Handle the prediction based on model output shape
            if len(pred.shape) > 0 and pred.shape[0] > 1:
                # Model outputs [with_mask, without_mask] probabilities
                mask_prob = pred[0]
                no_mask_prob = pred[1]
                
                # Determine if wearing mask based on model output
                is_wearing_mask = mask_prob > no_mask_prob
                
                # Apply label flipping if enabled
                if flip_labels:
                    is_wearing_mask = not is_wearing_mask
                
                if is_wearing_mask:
                    # Calculate the appropriate confidence value
                    confidence_value = mask_prob * 100 if not flip_labels else no_mask_prob * 100
                    label = f"Mask: {confidence_value:.1f}%"
                    color = (0, 255, 0)  # Green
                    frame_mask_count += 1
                else:
                    # Calculate the appropriate confidence value
                    confidence_value = no_mask_prob * 100 if not flip_labels else mask_prob * 100
                    label = f"No Mask: {confidence_value:.1f}%"
                    color = (0, 0, 255)  # Red
                    frame_no_mask_count += 1
            else:
                # Model outputs a single probability value (sigmoid)
                mask_prob = pred[0]
                
                # Determine if wearing mask based on model output
                is_wearing_mask = mask_prob > 0.5
                
                # Apply label flipping if enabled
                if flip_labels:
                    is_wearing_mask = not is_wearing_mask
                
                if is_wearing_mask:
                    confidence_value = mask_prob * 100 if not flip_labels else (1 - mask_prob) * 100
                    label = f"Mask: {confidence_value:.1f}%"
                    color = (0, 255, 0)  # Green
                    frame_mask_count += 1
                else:
                    confidence_value = (1 - mask_prob) * 100 if not flip_labels else mask_prob * 100
                    label = f"No Mask: {confidence_value:.1f}%"
                    color = (0, 0, 255)  # Red
                    frame_no_mask_count += 1
            
            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (startX, startY - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Update total counts
        mask_count += frame_mask_count
        no_mask_count += frame_no_mask_count
        
        # Add stats to frame
        output_frame = display_stats(frame, frame_mask_count, frame_no_mask_count, start_time)
        
        # Display frame if requested
        if args["display"]:
            cv2.imshow("Face Mask Detection", output_frame)
            key = cv2.waitKey(1) & 0xFF
            
            # Exit if 'q' pressed
            if key == ord("q"):
                break
        
        # Write frame to output video if requested
        if args["output"] is not None:
            # Initialize video writer if not already done
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 20,
                                        (output_frame.shape[1], output_frame.shape[0]), True)
            
            # Write frame
            writer.write(output_frame)
    
    # Print final stats
    elapsed_time = time.time() - start_time
    print("[INFO] Session summary:")
    print(f"  - Elapsed time: {elapsed_time:.1f} seconds")
    print(f"  - Processed frames: {frame_count}")
    print(f"  - Average FPS: {frame_count / elapsed_time:.2f}")
    print(f"  - With mask detections: {mask_count}")
    print(f"  - Without mask detections: {no_mask_count}")
    
    # Clean up
    if args.get("input"):
        vs.release()
    else:
        vs.stop()
        
    if writer is not None:
        writer.release()
        
    cv2.destroyAllWindows()
    print("[INFO] Program finished")

if __name__ == "__main__":
    main()