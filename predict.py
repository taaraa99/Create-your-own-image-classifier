import argparse
import torch
import json
import numpy as np
from PIL import Image
from torchvision import models
import torch.nn.functional as F
import os

# Prepares an image for the model by resizing, cropping, and normalizing it
def preprocess_image(img_path):
    """Preprocess the image to match the input format expected by the model."""
    
    # Open image and convert to RGB
    img = Image.open(img_path).convert("RGB")

    # Resize, keeping aspect ratio
    min_size = 256
    img.thumbnail((min_size, min_size))

    # Crop the center of the image to 224x224
    img_width, img_height = img.size
    left = (img_width - 224) / 2
    top = (img_height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))

    # Convert image to numpy array and normalize pixel values
    img_array = np.array(img) / 255.0
    mean_values = np.array([0.485, 0.456, 0.406])
    std_values = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean_values) / std_values

    # Rearrange dimensions (PyTorch expects channels first)
    img_array = img_array.transpose((2, 0, 1))

    return torch.tensor(img_array).float()

import torch
from torchvision import models
import os

def load_trained_model(checkpoint_path):
    """Loads a trained model from a checkpoint file, ensuring architecture and classifier match."""

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Load architecture correctly
    arch = checkpoint.get("arch", "vgg16")

    # Ensure correct architecture is loaded
    if arch == "vgg16":
        model = models.vgg16(weights="IMAGENET1K_V1")  # Ensure same VGG16 version
    elif arch == "vgg16_bn":
        model = models.vgg16_bn(weights="IMAGENET1K_V1")  # Use batch normalization if applicable
    elif arch == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
    else:
        raise ValueError(f"Unsupported architecture '{arch}' found in checkpoint.")

    # Freeze feature layers
    for param in model.parameters():
        param.requires_grad = False

    # Ensure classifier exists in checkpoint
    if "classifier" in checkpoint:
        model.classifier = checkpoint["classifier"]
    else:
        raise KeyError("Error: 'classifier' key not found in checkpoint. Make sure model was saved properly.")

    # Load model weights
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Ensure class-to-index mapping exists
    model.class_to_idx = checkpoint.get("class_to_idx", {})

    return model


# # Loads the model from a saved checkpoint
# def load_trained_model(checkpoint_path):
#     """Loads a trained model from a checkpoint file."""
    
#     # Load checkpoint data
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

#     # Load architecture
#     model_arch = checkpoint.get("arch", "vgg16")
#     if model_arch.startswith("vgg"):
#         model = getattr(models, model_arch)(pretrained=True)
#         model.classifier = checkpoint["classifier"]
#     elif model_arch == "resnet50":
#         model = models.resnet50(pretrained=True)
#         model.fc = checkpoint["classifier"]
#     else:
#         raise ValueError("Unsupported model type found in checkpoint")

#     # Load model state
#     model.load_state_dict(checkpoint["state_dict"])
#     model.class_to_idx = checkpoint["class_to_idx"]

#     return model

# Uses the model to make a prediction on an image
def make_prediction(img_path, model, top_k=5, device="cpu"):
    """Predict the top K most likely classes for an image using the trained model."""
    
    model.to(device)
    model.eval()

    # Process the image and add a batch dimension
    img_tensor = preprocess_image(img_path).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get probabilities
        log_probs = model(img_tensor)
        probs = torch.exp(log_probs)

        # Get top K predictions
        top_probs, top_classes = probs.topk(top_k, dim=1)
        top_probs = top_probs.cpu().numpy()[0]
        top_classes = top_classes.cpu().numpy()[0]

    # Convert class indices back to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_labels = [idx_to_class[class_id] for class_id in top_classes]

    return top_probs, class_labels

# Main function that handles command-line input
def main():
    parser = argparse.ArgumentParser(description="Classify an image using a trained deep learning model.")

    # Required arguments
    parser.add_argument("image_path", type=str, help="Path to the image to be classified.")
    parser.add_argument("checkpoint", type=str, help="Path to the trained model checkpoint.")

    # Optional arguments
    parser.add_argument("--top_k", type=int, default=5, help="Number of most probable classes to return.")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Path to JSON file mapping categories to names.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for prediction if available.")

    args = parser.parse_args()

    # Check if files exist before proceeding
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Error: Image file not found -> {args.image_path}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Error: Checkpoint file not found -> {args.checkpoint}")

    # Select device
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if args.gpu and not torch.cuda.is_available():
        print("Warning: GPU was requested but is not available. Using CPU instead.")

    # Load the trained model
    model = load_trained_model(args.checkpoint)

    # Make a prediction
    top_probs, top_classes = make_prediction(args.image_path, model, args.top_k, device)

    # Load category names if available
    if os.path.exists(args.category_names):
        with open(args.category_names, "r") as json_file:
            category_mapping = json.load(json_file)
        class_names = [category_mapping.get(str(cls), "Unknown") for cls in top_classes]
    else:
        print("Warning: Category names file not found. Displaying class numbers instead.")
        class_names = top_classes

    # Display prediction results
    print("\nPredictions:")
    for i, (prob, label) in enumerate(zip(top_probs, class_names), 1):
        print(f"{i}) {label} - {prob * 100:.2f}%")

    print("\nPrediction process completed.")

# Run the script
if __name__ == "__main__":
    main()
