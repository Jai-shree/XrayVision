import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import pandas as pd
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
                        'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_thickening',
                        'Cardiomegaly', 'Nodule Mass', 'Hernia', 'No Finding']

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = self.labels_frame.iloc[idx, 0]
        img_path = self.find_image(img_name)

        if img_path is None:
            image = Image.new('RGB', (1024, 1024), color='black')
            label = torch.zeros(len(self.classes))
        else:
            image = Image.open(img_path).convert('RGB')
            labels = self.labels_frame.iloc[idx, 1].split('|')
            label = torch.tensor([1 if c in labels else 0 for c in self.classes], dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, label

    def find_image(self, img_name):
        for i in range(1, 3):
            folder = f'images_{i:03d}'
            img_path = os.path.join(self.root_dir, folder, 'images', img_name)
            if os.path.exists(img_path):
                return img_path
        return None

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        #convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        #pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

csv_file = '/kaggle/input/data/Data_Entry_2017.csv'
data_dir = '/kaggle/input/data'
dataset = ChestXrayDataset(csv_file=csv_file, root_dir=data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=2)

# Set up model and training
model = SimpleCNN(num_classes=14)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            total += labels.numel()
            correct += (predicted == labels).sum().item()

    print(f'Accuracy after epoch {epoch+1}: {100 * correct / total:.2f}%')

print('Finished Training')

# Save the model state dictionary
model_path = 'chest_xray_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

import torch
from PIL import Image
from torchvision import transforms

# Define your model class (Same architecture you used before)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the model
model = SimpleCNN(num_classes=14)  # 14 classes as defined earlier
model.load_state_dict(torch.load('chest_xray_model.pth'))
model.eval()  # Set the model to evaluation mode

# Define image transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load and preprocess the image
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    image = transform(image)  # Apply the transformations
    image = image.unsqueeze(0)  # Add a batch dimension
    return image

# Function to predict from an X-ray image
def predict_xray(image_path, model):
    # Preprocess the image
    image = process_image(image_path)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        predicted = torch.sigmoid(output)  # Apply sigmoid to get probability between 0 and 1

        # Convert predictions to binary (threshold of 0.5)
        predicted_labels = (predicted > 0.3).float()

    return predicted_labels

# Example usage
image_path = '/kaggle/input/data/images_001/images/00000013_003.png'  # Path to the X-ray image
predicted_labels = predict_xray(image_path, model)
print(predicted_labels)
# Print predictions
classes = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
           'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_thickening',
           'Cardiomegaly', 'Nodule Mass', 'Hernia', 'No Finding']

for i, label in enumerate(predicted_labels[0]):
    if label == 1:
        print(f'{classes[i]} detected.')
    else:
        print(f'{classes[i]} not detected.')
