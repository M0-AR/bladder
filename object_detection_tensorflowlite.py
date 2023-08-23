import os
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter


def load_label_map(label_path):
    """
    Load labels from the specified path.

    Args:
    - label_path (str): Path to the label map file.

    Returns:
    - List of labels.
    """
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def process_frame(interpreter, frame, input_details, floating_model, width, height):
    """
    Process a video frame through the TFLite interpreter.

    Args:
    - interpreter (Interpreter): TensorFlow Lite model interpreter.
    - frame (numpy.ndarray): Video frame.
    - input_details (dict): Input tensor details from the interpreter.
    - floating_model (bool): Indicates if the model is floating point.
    - width (int): Desired width for frame preprocessing.
    - height (int): Desired height for frame preprocessing.

    Returns:
    - Processed input data for the TFLite model.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    return input_data


def main():
    # Input arguments
    MODEL_NAME = 'Sample_TFLite_model'
    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    VIDEO_NAME = 'C:/Users/md/Desktop/a.mp4'
    min_conf_threshold = 0.2

    # Paths
    CWD_PATH = os.getcwd()
    VIDEO_PATH = os.path.join(CWD_PATH, VIDEO_NAME)
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    # Load labels
    labels = load_label_map(PATH_TO_LABELS)

    # Load the Tensorflow Lite model
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width, height = input_details[0]['shape'][2], input_details[0]['shape'][1]
    floating_model = input_details[0]['dtype'] == np.float32

    # Detect if the model is TF1 or TF2 based on output layer name
    outname = output_details[0]['name']
    if 'StatefulPartitionedCall' in outname:
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    video = cv2.VideoCapture(VIDEO_PATH)
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print('Reached the end of the video!')
            break

        input_data = process_frame(interpreter, frame, input_details, floating_model, width, height)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        for i in range(len(scores)):
            if min_conf_threshold < scores[i] <= 1.0:
                ymin, xmin, ymax, xmax = [int(val) for val in (boxes[i] * [imH, imW, imH, imW])]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

                # Display label
                label = f"{labels[int(classes[i])]}: {int(scores[i] * 100)}%"
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow('Object detector', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
