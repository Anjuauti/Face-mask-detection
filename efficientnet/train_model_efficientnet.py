import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from imutils import paths
import io

def train_mask_detector(dataset_path, output_model_path, output_plot_path):
    """
    Train a binary mask detection model using the dataset with 'with_mask' and 'without_mask' folders.
    
    Args:
        dataset_path (str): Path to the dataset directory with class subdirectories
        output_model_path (str): Path where the trained model will be saved
        output_plot_path (str): Path where the training plot will be saved
    """
    # Initialize learning parameters
    INIT_LR = 1e-4
    EPOCHS = 20
    FINE_TUNE_EPOCHS = 10
    BS = 32
    IMG_SIZE = 224
    
    print("[INFO] Loading images from dataset...")
    image_paths = list(paths.list_images(dataset_path))
    
    # Check if both classes exist in the dataset
    class_dirs = set([os.path.basename(os.path.dirname(p)) for p in image_paths])
    print(f"[INFO] Found classes: {class_dirs}")
    
    if 'with_mask' not in class_dirs or 'without_mask' not in class_dirs:
        print("[ERROR] Dataset must contain 'with_mask' and 'without_mask' directories")
        return
    
    data = []
    labels = []
    
    # Load and preprocess images
    for image_path in image_paths:
        # Extract class label from directory name
        label = os.path.basename(os.path.dirname(image_path))
        
        # Skip incorrect_mask or other classes
        if label not in ['with_mask', 'without_mask']:
            print(f"[WARNING] Skipping image with unsupported class: {label}")
            continue
        
        # Load and preprocess image
        image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        image = img_to_array(image)
        image = preprocess_input(image)
        
        # Add to dataset
        data.append(image)
        labels.append(label)
    
    # Convert to numpy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    
    # Encode labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    
    # Print class information
    print(f"[INFO] Class names: {lb.classes_}")
    print(f"[INFO] 'with_mask' encoded as: {lb.transform(['with_mask'])[0]}")
    print(f"[INFO] 'without_mask' encoded as: {lb.transform(['without_mask'])[0]}")
    print(f"[INFO] Total images: {len(data)}")
    print(f"[INFO] Class distribution:")
    for i, class_name in enumerate(lb.classes_):
        count = np.sum(labels == i)
        print(f"  - {class_name}: {count}")
    
    # Split the data into training and testing sets
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)
    
    print(f"[INFO] Training on {len(trainX)} images, validating on {len(testX)} images")
    
    # Data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest")
    
    # Build model with EfficientNetB0 base
    print("[INFO] Building model with EfficientNetB0 architecture...")
    baseModel = EfficientNetB0(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    
    # Create the head of the model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(1, activation="sigmoid")(headModel)  # Binary classification
    
    # Attach head to the base model
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    # Freeze base model layers
    for layer in baseModel.layers:
        layer.trainable = False
    
    # Compile model with binary crossentropy
    print("[INFO] Compiling model...")
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    # Print model summary
    model.summary()
    
    # Train head of network
    print("[INFO] Training model head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)
    
    # Fine-tune the model
    print("[INFO] Fine-tuning model...")
    # Unfreeze the last 15 layers for fine-tuning
    for layer in baseModel.layers[-15:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    opt = Adam(learning_rate=INIT_LR/10)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    # Continue training
    H_finetune = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=FINE_TUNE_EPOCHS)
    
    # Evaluate the model
    print("[INFO] Evaluating model...")
    testPreds = model.predict(testX, batch_size=BS)
    testPreds = (testPreds > 0.5).astype(int)
    
    # Create classification report
    print("[INFO] Classification report:")
    cr = classification_report(testY, testPreds, target_names=lb.classes_)
    print(cr)
    
    # Show confusion matrix
    cm = confusion_matrix(testY, testPreds)
    print("Confusion Matrix:")
    print(cm)
    
    # Save model
    print(f"[INFO] Saving model to {output_model_path}...")
    # Use the .keras format instead of .h5 to avoid warnings
    model_path = output_model_path.replace('.h5', '.keras') if output_model_path.endswith('.h5') else output_model_path
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")
    
    # Save the class labels
    labels_path = os.path.splitext(output_model_path)[0] + "_labels.txt"
    with open(labels_path, "w") as f:
        for class_name in lb.classes_:
            f.write(f"{class_name}\n")
    print(f"[INFO] Class labels saved to {labels_path}")
    
    # Plot training results
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 8))
    
    # Plot initial training
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
    plt.title("Initial Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    
    # Plot fine-tuning
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, FINE_TUNE_EPOCHS), H_finetune.history["loss"], label="train_loss")
    plt.plot(np.arange(0, FINE_TUNE_EPOCHS), H_finetune.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, FINE_TUNE_EPOCHS), H_finetune.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, FINE_TUNE_EPOCHS), H_finetune.history["val_accuracy"], label="val_acc")
    plt.title("Fine-tuning Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"[INFO] Training plot saved to {output_plot_path}")
    
    # Save a text summary report using StringIO to capture model.summary() output
    # This avoids the UnicodeEncodeError
    report_path = os.path.splitext(output_model_path)[0] + "_report.txt"
    
    # First capture model summary
    summary_string = io.StringIO()
    model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
    model_summary = summary_string.getvalue()
    
    try:
        # Write the report with UTF-8 encoding
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("MASK DETECTION MODEL REPORT\n")
            f.write("==========================\n\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Total images: {len(data)}\n")
            f.write(f"Classes: {', '.join(lb.classes_)}\n\n")
            f.write("Model Architecture:\n")
            f.write(model_summary)
            f.write("\nTraining Parameters:\n")
            f.write(f"Initial learning rate: {INIT_LR}\n")
            f.write(f"Initial epochs: {EPOCHS}\n")
            f.write(f"Fine-tuning epochs: {FINE_TUNE_EPOCHS}\n")
            f.write(f"Batch size: {BS}\n\n")
            f.write("Classification Report:\n")
            f.write(cr)
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))
            f.write("\nLabel Encoding:\n")
            for i, class_name in enumerate(lb.classes_):
                f.write(f"  - {class_name}: {lb.transform([class_name])[0]}\n")
        
        print(f"[INFO] Model report saved to {report_path}")
    except Exception as e:
        print(f"[WARNING] Could not save model report: {str(e)}")
        # Try a simplified report without model summary as fallback
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("MASK DETECTION MODEL REPORT\n")
                f.write("==========================\n\n")
                f.write(f"Dataset: {dataset_path}\n")
                f.write(f"Total images: {len(data)}\n")
                f.write(f"Classes: {', '.join(lb.classes_)}\n\n")
                f.write("\nTraining Parameters:\n")
                f.write(f"Initial learning rate: {INIT_LR}\n")
                f.write(f"Initial epochs: {EPOCHS}\n")
                f.write(f"Fine-tuning epochs: {FINE_TUNE_EPOCHS}\n")
                f.write(f"Batch size: {BS}\n\n")
                f.write("Classification Report:\n")
                f.write(cr)
                f.write("\nConfusion Matrix:\n")
                f.write(str(cm))
            print(f"[INFO] Simplified model report saved to {report_path}")
        except Exception as e2:
            print(f"[ERROR] Could not save even simplified report: {str(e2)}")
    
    return model

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train binary mask detection model")
    parser.add_argument("-d", "--dataset", required=True,
                      help="path to input dataset with 'with_mask' and 'without_mask' folders")
    parser.add_argument("-m", "--model", type=str, default="mask_detector.keras",
                      help="path to output face mask detector model")
    parser.add_argument("-p", "--plot", type=str, default="training_plot.png",
                      help="path to output training plot")
    
    args = vars(parser.parse_args())
    
    # Train the model
    model = train_mask_detector(
        dataset_path=args["dataset"],
        output_model_path=args["model"],
        output_plot_path=args["plot"]
    )