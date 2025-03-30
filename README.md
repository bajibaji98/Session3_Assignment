# Session3_Assignment
Week 3 Assignment: Neural Network Models for Urban Scene Classification Using VS Code, Python, and GitHub Copilot
📌 Objective
In this assignment, you will:
✅ Implement a Convolutional Neural Network (CNN) for urban scene classification.
✅ Use MIT Places dataset (subset focusing on urban environments).
✅ Train, evaluate, and optimize your model using Batch Normalization and Dropout.
✅ Push all work to GitHub, submit a .zip of your repo, and record a video walkthrough of your project.

🛠️ Tools You Will Use
VS Code: Development environment
Python: Core programming language
GitHub Copilot: AI-assisted coding and debugging
GitHub: Version control and submission
PyTorch: Deep learning framework
NumPy, Matplotlib: Data handling and visualization
📁 Step 1: Project Setup and GitHub Integration
1. Create a New Project Folder
Open VS Code
Create a new folder for your project
Inside the folder, create a Python script: urban_scene_cnn.py
2. Initialize Git and Push to GitHub (Easiest Method)
Open the Source Control tab (Ctrl+Shift+G or Cmd+Shift+G).
Click "Initialize Repository".
Click "Publish to GitHub", sign in, and create a Public or Private repository.
To update your work:
Go to Source Control (Ctrl+Shift+G).
Type a commit message (e.g., "Initial commit").
Click "✔️ Commit" and then "Sync Changes".
✅ Your project is now on GitHub! 🎉

📦 Step 2: Install Required Libraries
Open a new terminal (Ctrl + ~ or Cmd + ~ on Mac) and install:

pip install torch torchvision numpy matplotlib opencv-python
✅ Push Your Changes to GitHub

Go to Source Control (Ctrl+Shift+G).
Type "Installed required libraries" as the commit message.
Click "✔️ Commit" and "Sync Changes".
📂 Step 3: Load and Prepare the Dataset (MIT Places Subset)
We will use PyTorch’s torchvision to load the MIT Places dataset (subset of urban categories).

1. Load the Dataset
Paste this into urban_scene_cnn.py:

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# Define dataset transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset from local directory (assumes dataset is downloaded)
dataset_path = "./MIT_Places_Urban_Subset"
dataset = ImageFolder(root=dataset_path, transform=transform)

# Split dataset into training, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Display a sample image
sample_image, sample_label = dataset[0]
plt.imshow(sample_image.permute(1, 2, 0))  # Convert tensor to image
plt.title(f"Sample Image - Class {dataset.classes[sample_label]}")
plt.axis("off")
plt.show()
✅ Push Your Changes to GitHub
Commit message: "Loaded and preprocessed MIT Places dataset"

🤖 Step 4: Build a Simple CNN Model
import torch.nn as nn
import torch.optim as optim

# Define a simple CNN architecture
class UrbanSceneCNN(nn.Module):
    def __init__(self, num_classes):
        super(UrbanSceneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 64 * 64, num_classes)  # Adjust based on image size

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x

# Initialize model
num_classes = len(dataset.classes)
model = UrbanSceneCNN(num_classes)
print(model)
✅ Push Your Changes to GitHub
Commit message: "Implemented CNN model for urban scene classification"

⚙️ Step 5: Train the CNN Model
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=5):
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}")

# Train model
train_model(model, train_loader, val_loader, optimizer, criterion)
✅ Push Your Changes to GitHub
Commit message: "Trained CNN model for urban scene classification"

📊 Step 6: Evaluate Model Performance
# Evaluate on test data
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot results
plt.bar(["Test Accuracy"], [test_accuracy])
plt.ylabel("Accuracy")
plt.title("CNN Model Performance")
plt.show()
✅ Push Your Changes to GitHub
Commit message: "Evaluated CNN model performance on test data"

📌 Submission Requirements
1. GitHub Repository
✅ Your GitHub repo must contain:

urban_scene_cnn.py (your Python script)
Training, evaluation, and results visualization
2. Download and Submit Your GitHub Repository
📌 Steps:

Go to your GitHub repository.
Click "Code" → "Download ZIP".
Upload the downloaded .zip file as part of your submission.
3. Video Walkthrough of Your Code
✅ Record a 5-7 minute video showing:

Your code in VS Code
How you trained the CNN
Comparison of results
Your findings and insights
📌 Submission Checklist ✅ Upload the ZIP file of your GitHub repo
✅ Submit a PowerPoint Presentation (6-7 slides with visuals)
✅ Submit a Video Walkthrough (5-7 minutes explaining your project)

🎉 Congratulations! You’ve Implemented a CNN for Urban Scene Classification! 🚀
✅ Remember: Everything must be pushed to GitHub before submission.
Let me know if you need any clarifications! 😊
