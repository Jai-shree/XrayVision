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
