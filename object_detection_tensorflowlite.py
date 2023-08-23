import numpy as np
import cv2
import tensorflow as tf

# Load the model
interpreter = tf.lite.Interpreter(model_path='Sample_TFLite_model/detect.tflite')
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read labels from labelmap
with open('Sample_TFLite_model/labelmap.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture('C:/Users/md/Desktop/a.mp4')


def preprocess(frame):
    # Adjust resizing according to your tflite model's input size
    frame = cv2.resize(frame, (320, 320))  # adjust this to match your model's expected input dimensions
    frame = frame / 255.0
    return frame.astype(np.float32)  # Cast to float32 here



def postprocess(frame, predictions):
    # Extract the class index with highest probability
    class_idx = np.argmax(predictions[0])
    class_label = labels[class_idx]

    # Overlay the label on the frame
    cv2.putText(frame, class_label, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    return frame


target_time_seconds = 47
frame_rate = 1
target_frame_number = target_time_seconds * frame_rate
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    processed_frame = preprocess(frame)

    # Set input tensor for interpreter and invoke it
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(processed_frame, axis=0))
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    output_frame = postprocess(frame, predictions)
    cv2.imshow('Object detection', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
