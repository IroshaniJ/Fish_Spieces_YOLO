import cv2
import yt_dlp
import torch
import streamlink

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True) 

# URL of the YouTube video
url = "https://www.youtube.com/watch?v=ssiZXK7mvJM"

# Extract video URL using yt-dlp
ydl_opts = {'format': 'best', 'noplaylist': True}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=False)
    video_url = info_dict['formats'][0]['url']

# Get the HLS stream using streamlink
streams = streamlink.streams(url)
stream = streams["best"]

# Open the stream using OpenCV
capture = cv2.VideoCapture(stream.url)

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Previous centroids for tracking object movement
prev_centroids = {}

# Set a threshold for minimum movement distance (adjust as needed)
min_distance = 5000 

while True:
    # Read a frame from the video
    ret, frame = capture.read()

    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area (adjust threshold as needed)
    min_area = 500  
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Calculate centroids of the current contours
    curr_centroids = {i: cv2.moments(cnt) for i, cnt in enumerate(filtered_contours)}

    # Check for movement exceeding a threshold distance
    for i, curr_centroid in curr_centroids.items():
        if curr_centroid['m00'] != 0: 
            cx, cy = int(curr_centroid['m10'] / curr_centroid['m00']), int(curr_centroid['m01'] / curr_centroid['m00'])

            if i in prev_centroids:
                prev_centroid = prev_centroids[i]
                if prev_centroid['m00'] != 0:
                    prev_cx, prev_cy = int(prev_centroid['m10'] / prev_centroid['m00']), int(prev_centroid['m01'] / prev_centroid['m00'])
                    distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5

                    if distance > min_distance:
                        # Object has moved significantly, proceed with YOLO detection
                        x, y, w, h = cv2.boundingRect(filtered_contours[i])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        roi = frame[y:y + h, x:x + w]
                        results = model(roi)

                        if len(results.render()) > 0:
                            annotated_roi = results.render()[0]
                            frame[y:y + h, x:x + w] = annotated_roi

    # Update previous centroids for the next iteration
    prev_centroids = curr_centroids.copy()

    # Display the frame
    cv2.imshow('YOLO Object Detection with Motion Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
capture.release()
cv2.destroyAllWindows()