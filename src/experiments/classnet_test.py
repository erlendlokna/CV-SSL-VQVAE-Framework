from src.models.classnet import ClassNet
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

#doesnt work quite yet


def to_vqs_dataloader(zqs, ys, batch_size=64):
    zqs = zqs.clone().to(torch.float)
    zqs_dataloader = TensorDataset(zqs, ys)
    return DataLoader(zqs_dataloader, batch_size=batch_size, shuffle=True)

def train_ClassNet(
        train_zqs,
        y_train,
        num_epochs=2000
    ):
    classifier = ClassNet(train_zqs.shape)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    zqs_train_dataloader = to_vqs_dataloader(train_zqs, torch.tensor(y_train))

    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0.0
        for inputs, targets in zqs_train_dataloader:
            optimizer.zero_grad()
            # Forward pass
            print(inputs.shape)
            predictions = classifier(inputs)
            # Compute the loss
            loss = criterion(predictions, targets)
            # Backpropagation
            loss.backward()
            # Update model parameters
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(zqs_train_dataloader)}")

    return classifier


    