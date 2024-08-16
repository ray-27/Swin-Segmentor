import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset


def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

# Assuming SwinUNet and other dependencies are already defined above
model = SwinUNet(in_channels=3, out_channels=20)  # 20 classes for Cityscapes
model = model.to(device='cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Run the training function
train_model(model, train_loader, criterion, optimizer)


