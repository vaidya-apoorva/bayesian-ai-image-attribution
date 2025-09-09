import os
import torch
from torchvision import transforms, models
from PIL import Image
import json

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names in the order used for training
class_names = ['Real', 'DALL-E', 'MidJourney', 'StableDiffusion']

# Image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load the trained model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("/mnt/ssd-data/vaidya/SReC/models/generator_classifier.pth", map_location=device))
model.to(device)
model.eval()

def get_classifier_probs(image_path):
    """Run the classifier on a single image and return class probabilities."""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return {cls: float(prob) for cls, prob in zip(class_names, probs)}

def main():
    root_dir = '/mnt/hdd-data/vaidya/dataset/JPEG90'
    ignore_folders = {'JPEG80', 'JPEG90', 'resized_images'}
    
    results = {}
    
    for folder_name in os.listdir(root_dir):
        if folder_name in ignore_folders:
            continue
        
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        print(f"Processing folder: {folder_name}")
        
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(folder_path, fname)
            try:
                probs = get_classifier_probs(img_path)
                results[img_path] = probs
                print(f"Processed: {img_path} -> {probs}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Save results to JSON file
    with open("/mnt/ssd-data/vaidya/SReC/results/classifier_results_JPEG_90.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("All images processed and results saved to classifier_results.json")

if __name__ == "__main__":
    main()
