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

while True:
    # Read a frame from the video
    ret, frame = capture.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Render the results on the frame
    annotated_frame = results.render()[0]

    # Display the frame
    cv2.imshow('YOLO Object Detection', annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
capture.release()
cv2.destroyAllWindows()