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

    # Draw bounding boxes around moving objects
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

        # Crop the region of interest (ROI) containing the moving object
        roi = frame[y:y+h, x:x+w]

        # Perform object detection on the ROI
        results = model(roi)

        # Render the results on the ROI
        if len(results.render()) > 0:
            annotated_roi = results.render()[0]

            # Overlay the annotated ROI back onto the original frame
            frame[y:y+h, x:x+w] = annotated_roi

    # Display the frame
    cv2.imshow('YOLO Object Detection with Motion Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
capture.release()
cv2.destroyAllWindows()