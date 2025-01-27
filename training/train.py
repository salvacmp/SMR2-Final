import os
from ultralytics import YOLO

def prepare_yolo_dataset(dataset_path, split_ratio=0.8):
    """
    Splits the dataset into training and validation sets.
    :param dataset_path: Path to the dataset directory containing images and labels.
    :param split_ratio: Ratio of training data. Remaining is used for validation.
    """
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")


    # Ensure directories exist
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Missing 'images' or 'labels' directory in {dataset_path}.")

    # Get all image files
    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    all_images.sort()  # Ensure consistent ordering

    # Split into train and val
    split_index = int(len(all_images) * split_ratio)
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    def write_split_file(image_list, split_name):
        split_file = os.path.join(dataset_path, f"{split_name}.txt")
        with open(split_file, 'w') as f:
            for img in image_list:
                f.write(f"{os.path.join(images_dir, img)}\n")
        print(f"{split_name} split saved to {split_file}")

    write_split_file(train_images, "train")
    write_split_file(val_images, "val")

def train_yolo(model_path, data_yaml, epochs=50, img_size=640, use_gpu=True):
    """
    Trains the YOLOv11 model.
    :param model_path: Path to save or load the YOLO model (e.g., 'yolov8n.pt').
    :param data_yaml: Path to the data YAML file describing the dataset.
    :param epochs: Number of training epochs.
    :param img_size: Image size for training.
    """
    # Load YOLO model
    model = YOLO(model_path)
    os.environ['WANDB_MODE'] = 'disabled'

    # Train model
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        name="yolov11_obb_model",
        device=0
    )
    print("Training complete!")

if __name__ == "__main__":
    # Define paths
    dataset_path = "./datasets"  # Update this path
    data_yaml_path = "./datasets/data.yaml"  # Update this path
    model_path = "./yolo11n-obb.pt"  # You can change to other YOLOv11-obb variants.

    # Prepare dataset (optional if already prepared)
    prepare_yolo_dataset(dataset_path)

    # Train YOLO model
    train_yolo(model_path, data_yaml_path, epochs=300, img_size=640)