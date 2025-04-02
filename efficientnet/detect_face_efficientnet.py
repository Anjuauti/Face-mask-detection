# USAGE
# python detect_mask_efficientnet.py --image images/pic1.jpeg

# Import necessary packages
import numpy as np
import argparse
import cv2
import os
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

def detect_mask_in_image():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")
    ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
        default="mask_detector.keras",
        help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # Load our serialized face detector model from disk
    print("[INFO] Loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # Load the face mask detector model from disk
    print("[INFO] Loading face mask detector model...")
    model = load_model(args["model"])

    # Load the input image from disk, clone it, and grab the image spatial dimensions
    if args["image"].lower().endswith(".jfif"):
      img = Image.open(args["image"])
      jpg_path = args["image"].replace(".jfif", ".jpg")
      img.save(jpg_path, "JPEG")
      args["image"] = jpg_path  # Update to use the converted file

    image = cv2.imread(args["image"])
    if image is None:
      print(f"[ERROR] Could not read image file: {args['image']}")
      return

# Resize image to a fixed size (e.g., 800x600)
    image = cv2.resize(image, (800, 600))

    if image is None:
      print(f"[ERROR] Could not read image file: {args['image']}")
      return
        
    orig = image.copy()
    (h, w) = image.shape[:2]

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    print("[INFO] Computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # Initialize counters for statistics
    face_count = 0
    mask_count = 0
    no_mask_count = 0

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > args["confidence"]:
            face_count += 1
            
            # Compute the (x, y)-coordinates of the bounding box for the object
            # New width and height after resizing
            (h, w) = image.shape[:2]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI, convert it from BGR to RGB channel ordering,
            # resize it to 224x224, and preprocess it (matching EfficientNetB0 input)
            face = image[startY:endY, startX:endX]
            
            # Check if face ROI is valid (not empty)
            if face.size == 0:
                print(f"[WARNING] Detected face region is empty. Skipping...")
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)  # Using EfficientNet's preprocessing
            face = np.expand_dims(face, axis=0)

            # Pass the face through the model to determine if the face has a mask or not
            # Model output is sigmoid (binary) in the EfficientNetB0 implementation
            pred = model.predict(face)[0]
            
            # For binary sigmoid output (from the training model)
            mask_prob = float(pred[0])
            withoutMask_prob = 1.0 - mask_prob
            
            label = "Mask" if mask_prob < 0.5 else "No Mask"
            
            if label == "Mask":
                mask_count += 1
                color = (0, 255, 0)  # Green for mask
                prob = mask_prob
            else:
                no_mask_count += 1
                color = (0, 0, 255)  # Red for no mask
                prob = withoutMask_prob

            # Include the probability in the label
            label = "{}: {:.2f}%".format(label, prob * 100)

            # Display the label and bounding box rectangle on the output frame
            cv2.putText(image, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # Add summary stats to the image
    stats = f"Summary: {face_count} faces, {mask_count} with mask, {no_mask_count} without mask"
    cv2.putText(image, stats, (10, h - 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the output image
    print("[INFO] Displaying result...")
    print(f"[INFO] {stats}")
    cv2.imshow("Output", image)
    
    # Save output image
    output_filename = f"output_{os.path.basename(args['image'])}"
    cv2.imwrite(output_filename, image)
    print(f"[INFO] Result saved to {output_filename}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_mask_in_image()