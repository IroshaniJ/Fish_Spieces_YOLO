import cv2
import yt_dlp
import torch
import streamlink
from ultralytics import YOLO
import argparse
from PIL import ImageColor, Image
import yaml
import random


# Read the YAML file
file_path = "GU_11:23:54_v2.yaml"
with open(file_path, 'r') as file:
    data = yaml.safe_load(file)

# Extract names and assign a color for each
names = data['names']

# Define fixed colors
color_list = ['red', 'green', 'blue', 'yellow']

# Create a dictionary assigning colors to species in order
colors = {name: color_list[i % len(color_list)] for i, name in enumerate(names)}

# Print or use the name-color dictionary
for name, color in colors.items():
    print(f"Species: {name}, Color: {color}")


def test_video(video_url, model):
    # Capture the video stream
    count = 0
    cap = cv2.VideoCapture(video_url)
    print("Processing video stream...")

    while cap.isOpened():
        count = count + 1
        ret, frame = cap.read()

        if not ret:
            print("Stream ended or connection failed")
            break
        if count % 3 == 0:
            # Detect objects in the frame using YOLO model
            results = model(frame)

            for result in results:
                objects = result.boxes.cpu().numpy()
                for j, box in enumerate(objects):
                    r = box.xyxy[0].astype(int)
                    confidence = box.conf[0].astype(float)
                    class_id = box.cls[0].astype(int)
                    label = names[class_id]
                    color = colors.get(label, "Color not assigned")
                    
                    # Convert color string to BGR tuple
                    rgb_color = ImageColor.getcolor(color, "RGB")
                    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                    text = f"{label} {confidence:.2f}"
                    
                    # Draw bounding box and label if confidence is above the threshold
                    if confidence > 0.20:
                        cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), bgr_color, 2)
                        font_scale = 0.5
                        thickness = 1
                        cv2.putText(frame, text, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, bgr_color, thickness)

            # Display the frame with YOLO detections
            cv2.imshow('YOLO Object Detection', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def main():
    # URL of the YouTube video stream
    url = "https://www.youtube.com/watch?v=ssiZXK7mvJM"

    # Extract video stream URL using yt-dlp
    ydl_opts = {'format': 'best', 'noplaylist': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_url = info_dict['formats'][0]['url']

    # Use streamlink to get the best HLS stream
    streams = streamlink.streams(url)
    stream = streams["best"]

    # Load the YOLO model
    model_path = "models/weights_v2/best.pt"
    model = YOLO(model_path)

    # Test the YOLO model on the real-time video stream
    test_video(stream.url, model)


if __name__ == "__main__":
    main()
