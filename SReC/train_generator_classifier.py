import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image

# 1. Define folder to class mapping
folder_to_class = {
    'coco': 'Real',
    'real': 'Real',
    'raise': 'Real',
    'dalle2': 'DALL-E',
    'dalle3': 'DALL-E',
    'midjourneyV5': 'MidJourney',
    'sdxl': 'StableDiffusion'
}

class_names = ['Real', 'DALL-E', 'MidJourney', 'StableDiffusion']

# 2. Define custom dataset that remaps folder names to grouped classes
class GroupedImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        
        new_samples = []
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
        new_targets = []
        
        for path, _ in self.samples:
            folder_name = os.path.basename(os.path.dirname(path))
            grouped_class = folder_to_class.get(folder_name)
            if grouped_class is not None:
                new_samples.append((path, class_to_idx[grouped_class]))
                new_targets.append(class_to_idx[grouped_class])
        
        self.samples = new_samples
        self.targets = new_targets
        self.classes = class_names
        self.class_to_idx = class_to_idx

# 3. Set up transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 4. Load dataset and split
data_dir = '/mnt/hdd-data/vaidya/dataset'
dataset = GroupedImageFolder(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 5. Load pretrained ResNet and modify final layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# 6. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 7. Training and evaluation loops
def train_one_epoch(model, dataloader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate(model, dataloader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# 8. Train the model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: Train loss={train_loss:.4f} acc={train_acc:.4f} | Val loss={val_loss:.4f} acc={val_acc:.4f}")

# 9. Save the trained model
torch.save(model.state_dict(), "generator_classifier.pth")
print("Model saved as generator_classifier.pth")
