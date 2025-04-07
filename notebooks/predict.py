import argparse
import torch
import json
from model import load_checkpoint
from utils import process_image

def predict(model, image_tensor, top_k, device):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to correct device
    
    with torch.no_grad():
        output = model(image_tensor)
    
    probabilities = torch.exp(output)  # Convert log-softmax output to probabilities
    top_probs, top_classes = probabilities.topk(top_k, dim=1)
    
    return top_probs.squeeze().tolist(), top_classes.squeeze().tolist()

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained network.")
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference if available")
    
    args = parser.parse_args()

    # Load the model
    model = load_checkpoint(args.checkpoint)
    
    # Select device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Process the image
    image = process_image(args.input)
    
    # Ensure the image is in the correct format (tensor and on device)
    image_tensor = torch.tensor(image, dtype=torch.float32)
    
    # Predict the class
    probs, classes = predict(model, image_tensor, args.top_k, device)

    # Map class indices to names if category_names JSON is provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name.get(str(cls), f"Class {cls}") for cls in classes]
    
    # Print predictions
    print("\nTop Predictions:")
    for i in range(len(classes)):
        print(f"Class: {classes[i]}, Probability: {probs[i]:.4f}")

if __name__ == "__main__":
    main()
