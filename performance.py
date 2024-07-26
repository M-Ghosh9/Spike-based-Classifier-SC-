#performance.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from model import ResNet18_1D  

def plot_performance_metrics(true_labels, predicted_labels, label_encoder, save_path='performance_metrics.png'):
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close(fig)

    # Classification Report
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_))

    # F1 Score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"F1 Score: {f1:.2f}")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(true_labels, predicted_labels, pos_label=1)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower right")
    plt.savefig('precision_recall_curve.png')
    plt.close(fig)

def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path='learning_curves.png'):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation loss values
    ax1.plot(epochs, train_losses, label='Training Loss')
    ax1.plot(epochs, val_losses, label='Validation Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot training & validation accuracy values
    ax2.plot(epochs, train_accuracies, label='Training Accuracy')
    ax2.plot(epochs, val_accuracies, label='Validation Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.savefig(save_path)
    plt.close(fig)

if __name__ == "__main__":
    # Load model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize LabelEncoder to encode labels
    label_encoder = LabelEncoder()

    # Load preprocessed signals and labels
    signals, labels = torch.load('signals.pt')
    signals = signals.to(device)
    
    # Encode labels
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_labels = torch.tensor(encoded_labels, dtype=torch.long).to(device)

    # Create dataset
    dataset = TensorDataset(signals, encoded_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)  # Reduced batch size

    # Determine number of classes
    num_classes = len(label_encoder.classes_)

    # Initialize model
    model = ResNet18_1D(num_classes).to(device)
    model.load_state_dict(torch.load('spike_classifier_signals_resnet.pth'))
    model.eval()

    # Predict on the entire dataset
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            torch.cuda.empty_cache()  # Clear CUDA cache periodically

    all_predictions = np.array(all_predictions)

    # Plot performance metrics
    plot_performance_metrics(encoded_labels.cpu().numpy(), all_predictions, label_encoder)

    # Load and plot learning curves
    train_losses = np.load('train_losses_signals_resnet.npy')
    val_losses = np.load('val_losses_signals_resnet.npy')
    train_accuracies = np.load('train_accuracies_signals_resnet.npy')
    val_accuracies = np.load('val_accuracies_signals_resnet.npy')

    plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)

