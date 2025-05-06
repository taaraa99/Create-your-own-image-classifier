import argparse
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import os

# loads and preprocesses the data, including transformations 
def data_transformation(data_directory):
    """Load and transform training, validation, and testing data."""
    
    train_dir = os.path.join(data_directory, "train")
    valid_dir = os.path.join(data_directory, "valid")

    # Make sure paths actually exist, otherwise it should throw an error
    for dir in [train_dir, valid_dir]:
        if not os.path.exists(dir):
            raise FileNotFoundError(f"Error: Directory not found -> {dir}")

    # Define transformations
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(root=valid_dir, transform=valid_transforms)

    # Dataloader for training and validation, with batch size and shuffle
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

    return train_loader, valid_loader, train_data.class_to_idx

# Loads the pre-trained model and adds a custom classifier
def setup_model(arch="vgg16", hidden_units=512):
    """Loads a pretrained model and replaces the classifier."""
    
    # Checks for architecture
    if arch.startswith("vgg"):
        model = getattr(models, arch)(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == "resnet50":
        model = models.resnet50(pretrained=True)
        input_size = model.fc.in_features
    else:
        raise ValueError("Unsupported architecture! Choose from: vgg11, vgg13, vgg16, vgg19, resnet50.")

    # Freeze feature extraction layers, don't want to train those
    for param in model.parameters():
        param.requires_grad = False

    # Defines a simple classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    # Attach the new classifier to the model
    if arch.startswith("vgg"):
        model.classifier = classifier
    elif arch == "resnet50":
        model.fc = classifier

    return model

# Training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=5, device="cuda"):
    """Train the model and validate it at each epoch."""
    
    model.to(device)
    print(f"\nStarting training on {device.upper()}...")

    for epoch in range(epochs):
        running_loss = 0
        model.train()  # set model to train mode

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation step, check model's performance
        model.eval()
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                valid_loss += loss.item()

                # Accuracy calculation
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train Loss: {running_loss/len(train_loader):.3f}.. "
              f"Validation Loss: {valid_loss/len(valid_loader):.3f}.. "
              f"Validation Accuracy: {accuracy/len(valid_loader)*100:.2f}%")

    return model



def save_checkpoint(model, save_dir, class_to_idx, arch, hidden_units, epochs, optimizer, lr):
    """Saves the trained model checkpoint with all required details."""

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{arch}.pth")

    checkpoint = {
        'arch': arch,  # Save architecture type
        'hidden_units': hidden_units,  # Store hidden units used in classifier
        'classifier': model.classifier,  # Save the classifier structure
        'state_dict': model.state_dict(),  # Model weights
        'class_to_idx': class_to_idx,  # Class label mapping
        'epochs': epochs,  # Save number of epochs
        'optimizer_state': optimizer.state_dict(),  # Save optimizer state
        'learning_rate': lr  # Store learning rate
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Model successfully saved at {checkpoint_path}")


# # Save the trained model checkpoint
# def save_checkpoint(model, save_dir, class_to_idx, arch, hidden_units, epochs, optimizer, lr):
#     """Saves trained model checkpoint."""

#     # Make sure save directory exists
#     os.makedirs(save_dir, exist_ok=True)
#     checkpoint_path = os.path.join(save_dir, f"checkpoint_{arch}.pth")

#     # Save all relevant model data
#     checkpoint = {
#         'arch': arch,
#         'hidden_units': hidden_units,
#         'state_dict': model.state_dict(),
#         'class_to_idx': class_to_idx,
#         'epochs': epochs,
#         'optimizer_state': optimizer.state_dict(),
#         'learning_rate': lr
#     }

#     # Save the checkpoint
#     torch.save(checkpoint, checkpoint_path)
#     print(f"\nModel saved to {checkpoint_path}")

# Parsing command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Train a deep learning model to classify flowers.")
    
    # Required arguments
    parser.add_argument("data_directory", help="Path to dataset folder (must contain 'train' and 'valid' subfolders).")

    # Optional arguments
    parser.add_argument("--save_dir", default="saved_models", help="Directory to save the trained model checkpoint.")
    parser.add_argument("--arch", default="vgg16", choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet50'], help="Model architecture.")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the classifier.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training if available.")

    args = parser.parse_args()

    # Check if GPU is available, otherwise use CPU
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if args.gpu and not torch.cuda.is_available():
        print("GPU was selected but it's not available, so training will be done on CPU.")

    # Load the dataset
    train_loader, valid_loader, class_to_idx = data_transformation(args.data_directory)

    # Set up model
    model = setup_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate) if args.arch.startswith("vgg") else optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    # Train the model
    model = train_model(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device)

    # Save trained model
    save_checkpoint(model, args.save_dir, class_to_idx, args.arch, args.hidden_units, args.epochs, optimizer, args.learning_rate)

if __name__ == "__main__":
    main()
