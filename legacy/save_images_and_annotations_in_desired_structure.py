import glob
import json
import os
from pathlib import Path

import cv2


def load_annotations(json_path: Path) -> dict:
    """
    Load annotations from the provided JSON file.

    Parameters:
        - json_path: Path to the JSON file.

    Returns:
        A dictionary containing the annotations.
    """
    with json_path.open("r") as f:
        json_data = json.load(f)
        return json_data["metadata"]

def extract_frames(video_path: Path, annotations: dict, frame_rate: int = 1):
    """
        Extracts frames from a given video based on provided annotations and crops the frames based on x, y coordinates.
        - If z has a single timestamp and xy has values, the frame at the timestamp is cropped using the xy values.
        - If z has two timestamps and xy is empty, frames between the timestamps are extracted without cropping.

        Parameters:
            - video_path: Path to the video.
            - annotations: A dictionary containing the time, label, and coordinates for frame extraction.
            - frame_rate: Extract every nth frame.

        Returns:
            A list of tuples where each tuple contains a (cropped) frame and its associated label.
        """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    frames = []
    try:
        for _, annotation_info in annotations.items():
            label = annotation_info["av"]["2"]
            xy_coords = annotation_info["xy"]

            # If only one timestamp and xy coordinates are present
            if len(annotation_info["z"]) == 1 and xy_coords:
                timestamp = annotation_info["z"][0]
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

                ret, frame = cap.read()
                if not ret:
                    # print(f"Warning: Unable to extract frame at {timestamp} seconds.")
                    continue

                # Assuming xy_coords are given in a format that can be converted to [xmin, ymin, xmax, ymax]
                x, y, width, height = map(float, xy_coords[1:5])

                xmin = int(x)
                ymin = int(y)
                xmax = int(x + width)
                ymax = int(y + height)

                print(f"xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")
                print(f"Frame shape: {frame.shape}")

                if xmin >= xmax or ymin >= ymax:
                    print("Invalid coordinates for cropping.")
                    continue

                if xmax > frame.shape[1] or ymax > frame.shape[0]:
                    print("Coordinates exceed frame dimensions.")
                    continue

                cropped_frame = frame[ymin:ymax, xmin:xmax]
                print(f"Cropped frame shape: {cropped_frame.shape}")

                # Check if the cropped frame is empty
                if cropped_frame.size == 0:
                    # print(f"Warning: Empty cropped frame at {timestamp} seconds.")
                    continue

                # frames.append((cropped_frame, label, timestamp))
                frames.append((cropped_frame, label, timestamp, (xmin, ymin, width, height)))
    finally:
        cap.release()
    return frames


if __name__ == "__main__":
    base_path = "E:/Mohamad/"
    frame_rate = 5

    dataset_path = "dataset"
    images_dir = os.path.join(dataset_path, "images")
    annotations_dir = os.path.join(dataset_path, "annotations")

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    for folder in os.listdir(base_path):
        current_folder_path = os.path.join(base_path, folder)

        if os.path.isdir(current_folder_path):
            # Search for the video and JSON file in the current folder
            video_files = glob.glob(os.path.join(current_folder_path, "*VIA*.mp4"))
            json_files = glob.glob(os.path.join(current_folder_path, "*.json"))

            if video_files and json_files:  # Check if both files are present
                video_path = Path(video_files[0])
                json_path = Path(json_files[0])

                annotations = load_annotations(json_path)
                extracted_frames = extract_frames(video_path, annotations, frame_rate)

                for i, (frame, label, timestamp, coords) in enumerate(extracted_frames):
                    filename = f"Frame_{timestamp}_{i + 1}.png"
                    filepath = os.path.join(images_dir, filename)
                    cv2.imwrite(filepath, frame)

                    # Save coordinates
                    annotation_filename = filename.replace(".png", ".txt")
                    annotation_filepath = os.path.join(annotations_dir, annotation_filename)

                    with open(annotation_filepath, "w") as f:
                        f.write(f"{coords[0]},{coords[1]},{coords[2]},{coords[3]}\n")

                    print(f"Saved {filename} to {images_dir}")
