# CNN-For-CIFAR10
## 🎯 Objective
Build and train a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. This implementation explores CNN architecture, feature extraction, model evaluation, and hyperparameter optimization.

## 📋 Requirements

### Software Dependencies
```bash
torch>=1.9.0
torchvision>=0.10.0
datasets>=2.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
numpy>=1.21.0
Pillow>=8.0.0
```

### Hardware Requirements
- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 4GB+ VRAM (CUDA-compatible)
- **Storage**: ~500MB for dataset and models

## 🚀 Installation

1. **Clone or download the implementation files**
2. **Install required packages**:
```bash
pip install torch torchvision datasets matplotlib seaborn scikit-learn numpy Pillow
```
3. **Verify PyTorch installation**:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 📁 File Structure
```
Question1_CNN/
│
├── cnn_cifar10.py          # Main implementation file
├── README.md               # This file
├── requirements.txt        # Package dependencies
└── outputs/                # Generated outputs (created automatically)
    ├── plots/              # Training curves and visualizations
    ├── models/             # Saved model checkpoints
    └── results/            # Performance metrics and tables
```

## 🏃‍♂️ Usage

### Basic Usage
```bash
python cnn_cifar10.py
```

### Advanced Usage with Custom Parameters
```python
from cnn_cifar10 import *

# Load data
train_dataset, test_dataset = load_cifar10_data()
train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, batch_size=64)

# Create custom model
model = CNN(num_filters=64, num_layers=5).to(device)

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs=20)
```

## 📊 Expected Outputs

### 1. Training Progress
```
Using device: cuda
Loading CIFAR-10 dataset...
Training for 10 epochs...
Epoch [1/10], Step [0/1563], Loss: 2.3156, Accuracy: 12.50%
Epoch [1/10], Step [100/1563], Loss: 2.1847, Accuracy: 23.45%
...
Epoch [10/10] - Loss: 0.8234, Accuracy: 78.45%
```

### 2. Performance Metrics
```
FINAL MODEL PERFORMANCE
==================================================
Test Accuracy: 85.67%
Precision: 0.8543
Recall: 0.8567
F1-Score: 0.8555
```

### 3. Hyperparameter Ablation Results
```
HYPERPARAMETER ABLATION STUDY RESULTS
========================================================================
Experiment      Value      Accuracy   Precision  Recall     F1-Score  
------------------------------------------------------------------------
Learning Rate   0.001      85.67      0.8543     0.8567     0.8555    
Learning Rate   0.01       82.34      0.8201     0.8234     0.8217    
Learning Rate   0.1        45.23      0.4456     0.4523     0.4489    
Batch Size      16         84.12      0.8389     0.8412     0.8400    
Batch Size      32         85.67      0.8543     0.8567     0.8555    
Batch Size      64         86.23      0.8601     0.8623     0.8612    
...
```

## 🧠 Implementation Details

### CNN Architecture
```python
CNN(
  (conv_layers): ModuleList(
    (0): Sequential(
      (0): Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1))
      (1): BatchNorm2d(32)
      (2): ReLU(inplace=True)
      (3): Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
      (4): BatchNorm2d(32)
      (5): ReLU(inplace=True)
      (6): MaxPool2d(kernel_size=2, stride=2)
    )
    ... (additional layers)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5)
    (1): Linear(in_features=2048, out_features=512)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5)
    (4): Linear(in_features=512, out_features=256)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=256, out_features=10)
  )
)
```

### Key Features
- **Dynamic Architecture**: Configurable number of layers and filters
- **Batch Normalization**: For stable training and faster convergence
- **Dropout Regularization**: Prevents overfitting
- **Data Augmentation**: Random horizontal flips and rotations
- **Feature Map Visualization**: Understanding learned representations

### CIFAR-10 Classes
```python
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]
```

## 📈 Results Interpretation

### Training Curves
- **Loss Curve**: Should decrease steadily, converging around 0.8-1.2
- **Accuracy Curve**: Should increase, reaching 75-90% on training data

### Confusion Matrix
- **Diagonal Values**: High values indicate good classification
- **Off-diagonal**: Common misclassifications (e.g., cat ↔ dog)

### Feature Maps
- **Layer 1**: Edge detection, basic shapes
- **Layer 2**: Textures, patterns
- **Layer 3+**: Complex features, object parts

### Hyperparameter Analysis
- **Learning Rate**: 0.001 typically optimal
- **Batch Size**: 32-64 usually best balance
- **Filters**: More filters = better features but slower training
- **Layers**: Deeper networks can learn complex patterns but risk overfitting

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce batch size
train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, batch_size=16)

# Or use CPU
device = torch.device('cpu')
```

#### 2. Multiprocessing Errors (Windows)
- Already fixed in implementation with `num_workers=0`
- If issues persist, run with `num_workers=0` in DataLoader

#### 3. Slow Training
```python
# Use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    print("Using CPU - training will be slower")
```

#### 4. Poor Performance
- Check learning rate (try 0.001, 0.0001)
- Verify data normalization
- Increase training epochs
- Add more data augmentation

### Performance Optimization
```python
# Enable mixed precision training (if supported)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(data)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 📝 Code Structure

### Main Components

1. **`CNN` Class**: The neural network architecture
2. **`load_cifar10_data()`**: Dataset loading and preprocessing
3. **`train_model()`**: Training loop with progress tracking
4. **`evaluate_model()`**: Model evaluation and metrics calculation
5. **`visualize_feature_maps()`**: Feature visualization
6. **`hyperparameter_ablation_study()`**: Systematic parameter testing

### Customization Options

```python
# Custom CNN architecture
model = CNN(
    num_filters=64,        # Starting number of filters
    num_layers=5          # Number of convolutional blocks
)

# Custom training parameters
train_losses, train_accuracies = train_model(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    num_epochs=20         # Adjust training duration
)
```

## 🎯 Assignment Deliverables

### Required Files
1. **Jupyter Notebook (.ipynb)** or **Python script (.py)**: `cnn_cifar10.py`
2. **PDF Report (LaTeX)**: Use results from this implementation
3. **GPT Prompts (.txt)**: Document all prompts used

### Report Sections
1. **Dataset Preparation**: CIFAR-10 loading and preprocessing
2. **Model Architecture**: CNN design and rationale
3. **Training Results**: Loss curves and performance metrics
4. **Feature Analysis**: What the CNN learned (feature maps)
5. **Hyperparameter Study**: Ablation results and optimal settings
6. **Conclusion**: Best configuration and performance analysis

## 🏆 Expected Performance

### Baseline Results
- **Test Accuracy**: 80-90%
- **Training Time**: 10-30 minutes (GPU), 2-4 hours (CPU)
- **Model Size**: ~6MB

### State-of-the-art Comparison
- **Simple CNN**: 80-85%
- **ResNet**: 90-95%
- **Vision Transformer**: 95%+

## 📚 References
- Original CIFAR-10 paper: Krizhevsky & Hinton (2009)
- CNN architectures: LeCun et al. (1998)
- Batch Normalization: Ioffe & Szegedy (2015)
- Data Augmentation: Simard et al. (2003)

## 💡 Tips for Success

1. **Start Simple**: Begin with basic architecture, then optimize
2. **Monitor Training**: Watch for overfitting/underfitting
3. **Visualize Results**: Feature maps reveal what model learned
4. **Document Everything**: Keep track of experiments
5. **Compare Results**: Use ablation study to find best configuration

---

**Happy Training! 🚀**

For questions or issues, refer to the troubleshooting section or check the PyTorch documentation.
