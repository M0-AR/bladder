"""
Thresholding using cv2.THRESH_BINARY:

When you apply normal binary thresholding (cv2.THRESH_BINARY):

Pixels with intensity values less than the threshold (in this case, 125) become black (0).
Pixels with intensity values greater than or equal to the threshold (125) become white (255).
So, considering the above:

Brighter areas (intensities >= 125) in the original grayscale image will appear as white in the thresholded image.
Darker areas (intensities < 125) will appear black. Thus, for normal binary thresholding, the regions of interest (
ROIs) in the thresholded image are the white areas, since they represent the brighter regions in the original
grayscale image.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_image(img, title="Image"):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


img_path = 'saved_frames/Normal/Frame_683.png'
img_path = 'saved_frames/NMIBC/Frame_1.png'
image = cv2.imread(img_path, cv2.IMREAD_COLOR)
display_image(image, "Original Image")


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
display_image(gray, "Grayscale Image")

_, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY) # 125 - 120
display_image(thresh, "Thresholded Image")

_, threshInv = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
display_image(threshInv, "Inverse Thresholded Image")


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
# closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
closed = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
display_image(closed, "After Morphological Closing")

closed = cv2.erode(closed, None, iterations=14)
display_image(closed, "After Erosion")

closed = cv2.dilate(closed, None, iterations=13)
display_image(closed, "After Dilation")


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

canny = auto_canny(closed)
display_image(canny, "Canny Edge Detection")


contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
display_image(image_with_contours, "Image with Contours")
