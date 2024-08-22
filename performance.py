#performance.py

import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import gc
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from model import ResNet18_1D
from matplotlib.backends.backend_pdf import PdfPages

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, pdf):
    logging.info("Plotting learning curves...")

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

    pdf.savefig(fig)
    plt.close(fig)

def plot_confusion_matrix_summary(true_labels, predicted_labels, label_encoder, pdf, top_n_classes=20):
    logging.info("Plotting confusion matrix summary...")

    cm = confusion_matrix(true_labels, predicted_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Summarize confusion matrix by top N most confused classes
    most_confused_pairs = np.dstack(np.unravel_index(np.argsort(-cm.ravel()), cm.shape))[0][:top_n_classes]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=False, fmt=".2f", cmap='Blues', ax=ax)
    for (i, j) in most_confused_pairs:
        ax.text(j + 0.5, i + 0.5, f'{cm[i, j]}', ha='center', va='center', color='red')
    ax.set_title(f'Confusion Matrix Summary (Top {top_n_classes} Confused Classes)')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    pdf.savefig(fig)
    plt.close(fig)

def plot_top_k_accuracies(true_labels, predicted_labels, model_logits, pdf, k_values=[1, 5, 10]):
    logging.info("Plotting top-k accuracies...")

    fig, ax = plt.subplots(figsize=(10, 6))
    top_k_acc = []

    for k in k_values:
        acc = top_k_accuracy_score(true_labels, model_logits, k=k, labels=np.arange(len(np.unique(true_labels))))
        top_k_acc.append(acc)
        ax.bar(f'Top-{k}', acc * 100, color='skyblue')

    ax.set_title('Top-K Accuracy')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)

    for i, v in enumerate(top_k_acc):
        ax.text(i, v * 100 + 1, f"{v * 100:.2f}%", ha='center')

    pdf.savefig(fig)
    plt.close(fig)

def plot_error_rate_distribution(true_labels, predicted_labels, pdf):
    logging.info("Plotting error rate distribution...")

    error_rates = 1 - np.diag(confusion_matrix(true_labels, predicted_labels)).astype(np.float) / np.bincount(true_labels)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(error_rates)), error_rates, color='orange')
    ax.set_title('Error Rate Distribution by Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Error Rate')

    pdf.savefig(fig)
    plt.close(fig)

def plot_classwise_metrics(true_labels, predicted_labels, label_encoder, pdf):
    logging.info("Plotting class-wise precision, recall, F1-score...")

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(np.arange(len(precision)), precision, label='Precision', color='blue')
    ax.plot(np.arange(len(recall)), recall, label='Recall', color='green')
    ax.plot(np.arange(len(f1)), f1, label='F1 Score', color='red')

    ax.set_title('Class-wise Precision, Recall, F1-Score')
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.legend()

    pdf.savefig(fig)
    plt.close(fig)

def evaluate_model_in_batches(model, dataloader, device):
    logging.info("Evaluating model in batches...")

    model.eval()
    all_predictions = []
    all_true_labels = []
    all_logits = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            logging.info(f"Processing batch {i+1}/{len(dataloader)}")
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())
            
            if i % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    logging.info("Finished evaluation.")
    return np.array(all_true_labels), np.array(all_predictions), np.array(all_logits)

if __name__ == "__main__":
    logging.info("Starting performance evaluation script...")

    # Load model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_encoder = LabelEncoder()

    logging.info("Loading preprocessed signals and labels...")
    signals, labels = torch.load('signals.pt')
    signals = signals.to(device)
    
    logging.info("Encoding labels...")
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_labels = torch.tensor(encoded_labels, dtype=torch.long).to(device)

    logging.info("Creating dataset and dataloader...")
    dataset = TensorDataset(signals, encoded_labels)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

    num_classes = len(label_encoder.classes_)
    logging.info(f"Number of classes: {num_classes}")

    logging.info("Initializing and loading the model...")
    model = ResNet18_1D(num_classes).to(device)
    model.load_state_dict(torch.load('spike_classifier_signals_resnet.pth'))

    logging.info("Evaluating the model...")
    true_labels, predicted_labels, model_logits = evaluate_model_in_batches(model, dataloader, device)

    logging.info("Generating performance report...")

    with PdfPages('model_performance_report.pdf') as pdf:
        plot_learning_curves(np.load('train_losses_signals_resnet.npy'),
                             np.load('val_losses_signals_resnet.npy'),
                             np.load('train_accuracies_signals_resnet.npy'),
                             np.load('val_accuracies_signals_resnet.npy'),
                             pdf)

        plot_confusion_matrix_summary(true_labels, predicted_labels, label_encoder, pdf, top_n_classes=20)
        plot_top_k_accuracies(true_labels, predicted_labels, model_logits, pdf, k_values=[1, 5, 10])
        plot_error_rate_distribution(true_labels, predicted_labels, pdf)
        plot_classwise_metrics(true_labels, predicted_labels, label_encoder, pdf)

    logging.info("Performance evaluation report saved as 'model_performance_report.pdf'.")


