import cv2
from ultralytics import YOLO
import argparse
from PIL import ImageColor, Image
import yaml
import random

# Read the YAML file
file_path = "fish_v3.yaml"
with open(file_path, 'r') as file:
    data = yaml.safe_load(file)

# Extract names and assign a color for each
# Assuming 'names' contains the species names
names = data['names']

# Define fixed colors
color_list = ['red', 'green', 'blue', 'yellow', "orange"]

# Create a dictionary assigning colors to species in order
colors = {name: color_list[i % len(color_list)] for i, name in enumerate(names)}

# Print or use the name-color dictionary
for name, color in colors.items():
    print(f"Species: {name}, Color: {color}")



# write a function to test the model on a each frame of the video and return the output video
# input: video path, output video path, model path
# output: output video path
# load the model

def test_video(video_path, output_path, model):

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    # save as .mp4
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # split the file name from / and then split the file name from .
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print("Processing video...", video_path)

    # loop through the video frames
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
       
        if not ret:
            break


        # detect the objects in the images
        #resized = cv2.resize(image, (640, 640))
        results = model(frame)
        for result in results:
                objects = result.boxes.cpu().numpy()
                for j, box in enumerate(objects):
                    r = box.xyxy[0].astype(int)
                    # scale bounding box coordinates to the original image size
                    confidence = box.conf[0].astype(float)
                    class_id = box.cls[0].astype(int)
                    label = names[class_id]
                    color = colors.get(label, "Color not assigned")
                    # Convert the color string to a BGR tuple
                    rgb_color = ImageColor.getcolor(color, "RGB")
                    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                    text = f"{label} {confidence:.2f}"
                    
                    if (class_id==6 and confidence > 0.05) or confidence > 0.45:
                        cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), bgr_color, 2)
                        font_scale = 0.5  # Smaller font size
                        thickness = 1     # Adjust thickness as needed
                        cv2.putText(frame, text, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, bgr_color, thickness)
                        file_name = './test_images/'+ str(count)+'.jpg'
                        cv2.imwrite(file_name,frame)

                    print(f"Detected {label} with confidence {confidence:.2f}")
                    count = count + 1
        
        # write the frame to the output video
        out.write(frame)


    # release the video
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_path

def main():
    video_path = "./videos/clip4.mp4"
    output_path = "./output_videos/output4_fintunedv3.avi"
    model_path = "models/weights_v3/best.pt"
    model = YOLO(model_path)
    test_video(video_path, output_path, model)


if __name__ == "__main__":
    main()
