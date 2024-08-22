#model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

# Model Definition with updated dropout probability
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_prob=0.6):  # Updated dropout_prob to 0.6
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet1D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.mean(dim=2)  # Global average pooling
        out = self.linear(out)
        return out

def ResNet18_1D(num_classes):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes)

def compute_class_weights(labels):
    class_sample_count = np.unique(labels, return_counts=True)[1]
    class_weights = 1. / class_sample_count
    return class_weights

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return loss, accuracy

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved as '{filename}'")

def load_checkpoint(filename, model=None, optimizer=None):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_accuracy = checkpoint['best_val_accuracy']
        return epoch, best_val_accuracy
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, 0

def plot_class_distributions(labels, class_weights, filename='class_distributions.png'):
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_weights_dict = dict(enumerate(class_weights))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(unique_labels, counts, color='skyblue')
    plt.xlabel('Class Labels')
    plt.ylabel('Counts')
    plt.title('Class Distribution Before Weighting')

    plt.subplot(1, 2, 2)
    weighted_counts = counts * np.array([class_weights_dict[label] for label in unique_labels])
    plt.bar(unique_labels, weighted_counts, color='lightcoral')
    plt.xlabel('Class Labels')
    plt.ylabel('Weighted Counts')
    plt.title('Class Distribution After Weighting')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load preprocessed signals and labels
        print("Loading data...")
        signals, labels = torch.load('signals.pt')
        signals = signals.to(device)
        
        # Encode labels
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        encoded_labels = torch.tensor(encoded_labels, dtype=torch.long).to(device)

        # Compute class weights
        print("Computing class weights...")
        class_weights = compute_class_weights(encoded_labels.cpu().numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        # Plot class distributions
        print("Plotting class distributions...")
        plot_class_distributions(
            labels=encoded_labels.cpu().numpy(),
            class_weights=class_weights.cpu().numpy(),
            filename='class_distributions.png'
        )

        # Create dataset
        dataset = TensorDataset(signals, encoded_labels)

        # Split dataset into training and validation sets (80-20)
        print("Splitting dataset...")
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Initialize model, criterion, and optimizer
        print("Initializing model, criterion, and optimizer...")
        model = ResNet18_1D(num_classes=len(np.unique(encoded_labels.cpu().numpy()))).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-4)  # Lowered learning rate

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

        # Load checkpoint if available
        start_epoch, best_val_accuracy = load_checkpoint('checkpoint_best.pth', model, optimizer)
        
        # Training loop
        num_epochs = 5000
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        print("Starting training loop...")
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(train_dataloader)
            train_accuracy = 100. * correct / total

            val_loss, val_accuracy = evaluate_model(model, val_dataloader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            # Save checkpoint if validation accuracy improves
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_accuracy': best_val_accuracy
                }, filename='checkpoint_best.pth')

            # Step the learning rate scheduler
            scheduler.step()

        # Final save
        print("Saving final model...")
        torch.save(model.state_dict(), 'spike_classifier_signals_resnet.pth')
        print("Final model saved as 'spike_classifier_signals_resnet.pth'")
        
        # Save training and validation curves
        print("Saving training and validation curves...")
        np.save('train_losses_signals_resnet.npy', train_losses)
        np.save('val_losses_signals_resnet.npy', val_losses)
        np.save('train_accuracies_signals_resnet.npy', train_accuracies)
        np.save('val_accuracies_signals_resnet.npy', val_accuracies)

    except Exception as e:
        print(f"An error occurred: {e}")
