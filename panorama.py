import cv2
def get_total_frames(video_path):
    """
    Returns the total number of frames in a video.

    Parameters:
    - video_path (str): The path to the input video.

    Returns:
    - int: Total number of frames. Returns -1 if unable to open video.

    Example usage:
    total_frames = get_total_frames("path_to_video.mp4")
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return -1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return total_frames


# Test the function
if __name__ == "__main__":
    video_path = 'C:/Users/md/Desktop/a.mp4'
    print(f"Total frames in the video: {get_total_frames(video_path)}")



import cv2
import numpy as np

def compute_homography(src, dst):
    """
    Compute the homography matrix from the source to the destination.
    """
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(src, None)
    keypoints2, descriptors2 = sift.detectAndCompute(dst, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M

def stitch_images(video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return None

    ret, base_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        return None

    scale_factor = 0.5
    base_frame = cv2.resize(base_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    stitched_image = base_frame
    frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_rate == 0:
            H = compute_homography(frame, stitched_image)
            h, w, _ = stitched_image.shape
            stitched_image = cv2.warpPerspective(frame, H, (w*2, h))

    cap.release()
    return stitched_image

if __name__ == "__main__":
    video_path = 'C:/Users/md/Desktop/a.mp4'

    panorama = stitch_images(video_path, frame_rate=1000)

    if panorama is not None:
        cv2.imshow("Stitched Image", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()