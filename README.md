🧠 Spike-based Classifier SC

Spike-based classification using a ResNet-inspired deep learning architecture tailored for time-series neural spike data. This project addresses the challenge of multi-class classification in imbalanced datasets, leveraging deep residual learning to improve accuracy and robustness.

📌 Overview
This repository implements a deep neural network for classifying neural spike waveforms. It adapts ResNet-style skip connections to handle the temporal dynamics of spike signals and includes preprocessing, training, and evaluation pipelines.

🏗️ Project Structure
``` plaintext
Spike-based-Classifier-SC-/
├── data_loader.py         # Data loading and preprocessing
├── model.py               # ResNet-inspired model architecture
├── performance.py         # Training, validation, and performance metrics
├── README.md              # Project documentation
└── TrainingData/          # Directory for input spike waveform data
```


📊 Features
- ✅ ResNet-inspired architecture for time-series classification
- ✅ Handles multi-class imbalance
- ✅ Visualizes training/validation accuracy and loss
- ✅ Spike waveform preprocessing and normalization
- ✅ Modular and extensible codebase

🧪 Requirements
``` plaintext
- Python 3.7+
- PyTorch
- NumPy
- scikit-learn
- matplotlib
Install dependencies:
pip install -r requirements.txt
```



📁 Data Format
- Input data should be organized in subdirectories under TrainingData/, where each subdirectory represents a class label.
- Each .txt file contains a single spike waveform (one sample per line).
Example structure:
``` plaintext
TrainingData/
├── Class_0/
│   ├── spike1.txt
│   └── spike2.txt
├── Class_1/
│   ├── spike1.txt
│   └── spike2.txt
...
```


🚀 Getting Started
- Prepare your data in the format above.
- Run the main script:
``` plaintext
python performance.py
```

This will:
- Load and preprocess the data
- Train the model
- Evaluate performance
- Display accuracy/loss plots

📈 Output
- Training and validation accuracy/loss curves
- Spike classification performance per class
- Label distribution and spike counts

🧠 Model Architecture
The model is inspired by ResNet, adapted for 1D time-series data:
- 1D convolutional layers
- Batch normalization
- ReLU activations
- Residual skip connections
- Fully connected classification head

📉 Handling Class Imbalance
The pipeline includes:
- Label distribution checks
- Spike count per class
- Optional oversampling or class weighting (can be added)


Overall Simple Process Pipeline-
![image](https://github.com/user-attachments/assets/01933001-c017-4e07-9f22-0fa69c5a0184)


ResNet Inspired Architecture for Time-Series Data-
![image](https://github.com/user-attachments/assets/aa059fd9-103e-474f-8cdc-5f89bfa42365)


Dealing with multi-class calss imbalance- 
![image](https://github.com/user-attachments/assets/e631c55a-639a-4f53-b68c-5a9c7822cc74)

Results-
Training and Validation Accuracy-
![image](https://github.com/user-attachments/assets/8249a27b-4c3c-40ff-8362-777812c5871b)

Training and Validation Loss- 
![image](https://github.com/user-attachments/assets/460a71c7-eb8b-429d-98f4-8f00d087cfa8)

Spikes Output-
![image](https://github.com/user-attachments/assets/b09f9cba-5911-4734-bfd2-988f42afefa2)

📌 To Do
- [ ] Add support for test-time augmentation
- [ ] Integrate confusion matrix visualization
- [ ] Export trained model for deployment

