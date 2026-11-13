import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt


# Functions for image processing and display
def display_image(img, title="Image"):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

# Load deep learning model
model = load_model('bladder_types.h5')

# Pre-processing for deep learning prediction
def preprocess(frame):
    frame = cv2.resize(frame, (1730, 1080))
    frame = frame / 255.0
    return frame

# Post-process the predictions
def postprocess(frame, predictions):
    class_idx = np.argmax(predictions[0])
    labels = ['NMIBC', 'Normal', 'Ta']
    class_label = labels[class_idx]
    cv2.putText(frame, class_label, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    return frame

# Image processing for tumor region detection
img_path = 'saved_frames/NMIBC/Frame_1.png'
img_path = 'saved_frames/Normal/Frame_683.png'
image = cv2.imread(img_path, cv2.IMREAD_COLOR)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
_, threshInv = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
# closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
closed = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, None, iterations=14)
closed = cv2.dilate(closed, None, iterations=13)
canny = auto_canny(closed)

contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

# Make prediction using your model
processed_image = preprocess(image_with_contours)
predictions = model.predict(np.expand_dims(processed_image, axis=0))
output_image = postprocess(image_with_contours, predictions)

# Display the processed image with contours and prediction label
display_image(output_image, "Processed Image with Prediction")
