# Fish Species Identification in Video Stream Using YOLO Model

This project uses a finetuned YOLO model to identify different fish species living in Oslofjord including fish spieces such as:

- **Ctenolabrus rupestris** (Goldsinny Wrasse)
- **Gadus morhua** (Atlantic cod)
- **Ascidiacea** (Sea Squirt)
- **Asteroidea** (Starfish)


## Usage

To detect fish using the model on a video stream, run the following command:
```bash
python3 test.py
```
You must modify the file to add video_path and output_path


To detect fish using the model on a live youtube video stream, run the following command:

```bash
python3 test_live.py
```
You must modify the file to add link to youtube video (url)

### Key Features:
- Uses **YOLO** object detection model. The model is finetuned on the model at https://zenodo.org/records/10932673 created by https://dto-bioflow.eu/
- Trained to recognize **four distinct species** in video streams.

## Based on:
https://zenodo.org/records/10932673


You might want to install ffmpeg for the youtube live stream part.

```bash
brew install ffmpeg
```
