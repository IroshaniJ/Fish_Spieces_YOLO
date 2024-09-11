from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

# Initialize a Weights & Biases run
wandb.init(project="ultralytics", job_type="training")

# Load a model
model = YOLO("best.pt")  # load a pretrained model (recommended for training)

# Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Train the model with 2 GPUs
results = model.train(project="ultralytics", data="GU_11:23:54.yaml", epochs=100, imgsz=640, device=[0, 1], batch=16)

# Validate the Model
model.val()

# Perform Inference and Log Results
model(["./test/images/163_jpg.rf.668c05290da5383638c8209c1ab79010.jpg"])

# Finalize the W&B Run
wandb.finish()