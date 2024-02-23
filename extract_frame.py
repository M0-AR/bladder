import os
import cv2
import json
from pathlib import Path
import glob


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
                frames.append((frame, label, timestamp))

            # # If two timestamps and no xy coordinates
            # elif len(annotation_info["z"]) == 2:
            #     start_time, end_time = annotation_info["z"]
            #     cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            #     frame_count = 0
            #
            #     while cap.get(cv2.CAP_PROP_POS_MSEC) <= end_time * 1000:
            #         ret, frame = cap.read()
            #         if not ret:
            #             # print(f"Warning: Unable to extract frame at {timestamp} seconds.")
            #             break
            #
            #         frame_count += 1
            #         if frame_count % frame_rate == 0:
            #             frames.append((frame, label))
    finally:
        cap.release()

    return frames


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


if __name__ == "__main__":
    base_path = "E:/Mohamad/"
    frame_rate = 5  # Extract every 5th frame (Note: You mentioned 100th in the comment, but used 5 in code)

    # Iterate over each directory inside base_path
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

                # Modify save_dir to reflect the current folder name
                save_dir = os.path.join("desire-frames", folder)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                for i, (frame, label, timestamp) in enumerate(extracted_frames):
                    label_dir = os.path.join(save_dir, label)

                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir)

                    filename = f"Frame_{timestamp}_{i + 1}.png"
                    filepath = os.path.join(label_dir, filename)
                    cv2.imwrite(filepath, frame)

                    print(f"Saved {filename} to {label_dir}")

"""
if __name__ == "__main__":
    # video_path = Path("C:/Users/md/Desktop/a.mp4")
    # json_path = Path("C:/Users/md/Desktop/a.json")
    video_path = Path("E:/Mohamad/(195) 220240-1085 T/220240-1085 - VIA.mp4")
    json_path = Path("E:/Mohamad/(195) 220240-1085 T/195 - via_project_20Jul2022_12h43m49s.json")
    frame_rate = 5  # Extract every 100th frame

    annotations = load_annotations(json_path)
    extracted_frames = extract_frames(video_path, annotations, frame_rate)

    # Define the directory where you want to save the frames
    save_dir = "frames-195"

    # Create the main directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (frame, label) in enumerate(extracted_frames):
        # Define the sub-directory based on the label
        label_dir = os.path.join(save_dir, label)

        # Create the sub-directory for the label if it doesn't exist
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Define the filename based on the frame number and label
        filename = f"Frame_{i + 1}.png"  # or ".jpg"
        filepath = os.path.join(label_dir, filename)

        # Save the frame to the specified directory
        cv2.imwrite(filepath, frame)

        print(f"Saved {filename} to {label_dir}")
    
   """
    # # Create a named window with a specific property
    # cv2.namedWindow("My Window", cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL for resizable window
    #
    # # Set the desired window size
    # cv2.resizeWindow("My Window", 600, 400)
    #
    # for i, (frame, label) in enumerate(extracted_frames):
    #     cv2.imshow(f"Frame {i + 1} - {label}", frame)
    #
    #     # Wait for a key event with a delay (0 means wait indefinitely until a key is pressed)
    #     key = cv2.waitKey(0)
    #
    #     # Check if the 'q' key is pressed to quit the loop
    #     if key == ord('q'):
    #         break
    #
    # # Destroy all created windows
    # cv2.destroyAllWindows()