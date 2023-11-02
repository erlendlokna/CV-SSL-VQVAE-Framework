from src.models.classnet import ClassNet, CNNClassNet, check_type
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
#doesnt work quite yet


def to_vqs_dataloader(zqs, ys, batch_size=64):
    if isinstance(zqs, np.ndarray):
        zqs = torch.tensor(zqs, dtype=torch.float)
    else:
        zqs = zqs.clone().to(torch.float)
    
    zqs_dataloader = TensorDataset(zqs, ys)
    return DataLoader(zqs_dataloader, batch_size=batch_size, shuffle=True)

def train_CNNClassNet(
        train_zqs,
        y_train,
        num_epochs=2000,
        silent=True
    ):
    assert len(train_zqs.shape) == 4, "please provide zqs_train with length 4"
    input_channels = int(train_zqs.shape[1])
    input_height = int(train_zqs.shape[2])
    input_width = int(train_zqs.shape[3])
    

    classifier = CNNClassNet(input_channels, input_height, input_width, num_classes=len(np.unique(y_train)))
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    zqs_train_dataloader = to_vqs_dataloader(train_zqs, torch.tensor(y_train))

    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0.0
        for inputs, targets in zqs_train_dataloader:
            optimizer.zero_grad()
            # Forward pass
            predictions = classifier(inputs)
            # Compute the loss
            loss = criterion(predictions, targets)
            # Backpropagation
            loss.backward()
            # Update model parameters
            optimizer.step()
            total_loss += loss.item()

        if not silent: print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(zqs_train_dataloader)}")
        if total_loss / len(zqs_train_dataloader) < 0.0005: 
            print("stopping training")
            break
    print(f"final epoch loss: {total_loss / len(zqs_train_dataloader)}")
    return classifier



def train_ClassNet(
    train_zqs,
    y_train,
    num_epochs=2000,
    learning_rate=0.001,
    silent=True
):
    train_zqs = check_type(train_zqs)
    
    input_length = train_zqs.shape[1]
    num_classes = len(np.unique(y_train))

    model = ClassNet(input_length, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    zqs_train_dataloader = to_vqs_dataloader(train_zqs, torch.tensor(y_train))

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0

        for inputs, targets in zqs_train_dataloader:
            optimizer.zero_grad()
            # Forward pass
            predictions = model(inputs)
            # Compute the loss
            loss = criterion(predictions, targets)
            # Backpropagation
            loss.backward()
            # Update model parameters
            optimizer.step()
            total_loss += loss.item()

        if not silent: print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(zqs_train_dataloader)}")
        if total_loss / len(zqs_train_dataloader) < 0.0005: 
            print("stopping training")
            break
    
    print(f"final epoch loss: {total_loss / len(zqs_train_dataloader)}")
    return model