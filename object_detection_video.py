import math

from keras.models import load_model
import cv2
import numpy as np

# Load the model
model = load_model('bladder_types.h5')

# Open the video file
cap = cv2.VideoCapture('C:/Users/md/Desktop/a.mp4')

def preprocess(frame):
    # Resize the frame
    frame = cv2.resize(frame, (1730, 1080))  # your model's expected input size

    # Normalize pixel values (this assumes your images are in the format of BGR)
    frame = frame / 255.0  # if you used this kind of normalization during training

    return frame

def postprocess(frame, predictions):
    # Get the index of the class with the highest predicted probability
    class_idx = np.argmax(predictions[0])

    labels = ['NMIBC', 'Normal', 'Ta']
    # Convert the class index to its corresponding name
    class_label = labels[class_idx]
    # class_label = str(class_idx)  # Convert class_idx to string

    # Overlay the class label on the frame
    cv2.putText(frame, class_label, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    return frame

# Determine the frame number to skip to
target_time_seconds = 47
frame_rate = 1
target_frame_number = target_time_seconds * frame_rate

# Skip to the target frame
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)

while True:
    # Read the video frame by frame
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame to be suitable for your model
    # This will depend on how you preprocessed your images during training.
    processed_frame = preprocess(frame)

    # Make prediction using your model
    predictions = model.predict(np.expand_dims(processed_frame, axis=0))

    # Postprocess predictions and draw bounding boxes, labels etc. on the frame
    # This will also depend on the structure of your model's output.
    output_frame = postprocess(frame, predictions)

    # Display the frame
    cv2.imshow('Object detection', output_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
