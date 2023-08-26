import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt

# Load the Keras model
model = load_model('bladder_types.h5')


def display_image(img, title="Image"):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def preprocess_for_contouring(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    _, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations=14)
    closed = cv2.dilate(closed, None, iterations=13)

    canny = cv2.Canny(closed, 50, 150)

    contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    return frame


def preprocess(frame):
    frame = cv2.resize(frame, (1730, 1080))
    frame = frame / 255.0
    return frame


def postprocess(frame, predictions):
    class_idx = np.argmax(predictions[0])
    labels = ['NMIBC', 'Normal', 'Ta']
    class_label = labels[class_idx]

    cv2.putText(frame, 'Suspected Region' if class_label != 'Normal' else class_label, (5, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    return frame


cap = cv2.VideoCapture('C:/Users/md/Desktop/a.mp4')

target_time_seconds = 0
frame_rate = 1
target_frame_number = target_time_seconds * frame_rate

cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Contour-based tumor detection
    frame_with_contours = preprocess_for_contouring(frame)

    # Preprocess the frame for Keras model classification
    processed_frame = preprocess(frame_with_contours)

    # Make predictions using the Keras model
    predictions = model.predict(np.expand_dims(processed_frame, axis=0))
    output_frame = postprocess(frame_with_contours, predictions)

    # Display the frame
    cv2.imshow('Tumor Detection and Classification', output_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
