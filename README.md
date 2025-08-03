# 🎨 Generative Adversarial Network (GAN) for MNIST Handwritten Digits

This project implements a **Generative Adversarial Network (GAN)** using PyTorch to generate synthetic handwritten digits resembling the MNIST dataset. The GAN consists of a Generator and a Discriminator, trained adversarially to produce realistic images of digits. This README provides an overview of the project, its methodology, achievements, and instructions for running the code.

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training Process](#training-process)
  - [Visualization](#visualization)
- [Achievements](#achievements)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

The goal of this project is to build and train a GAN to generate synthetic images of handwritten digits that mimic the style and quality of the MNIST dataset. The GAN framework leverages a **Generator** to create fake images from random noise and a **Discriminator** to distinguish between real and fake images. Through adversarial training, the Generator improves its ability to produce realistic digits, while the Discriminator enhances its ability to differentiate real from fake.

This project demonstrates the application of deep learning techniques for generative modeling, showcasing the power of GANs in creating high-quality synthetic data.

## 🔧 Methodology

### 📊 Data Preprocessing

The MNIST dataset, consisting of 28x28 grayscale images of handwritten digits (0-9), is used for training. The preprocessing steps include:

- **Loading the Dataset**: The dataset is loaded using `torchvision.datasets.MNIST`, with the training set automatically downloaded if not present.
- **Transformation**: Images are transformed using `torchvision.transforms`:
  - Converted to tensors using `ToTensor()`
  - Normalized to the range [-1, 1] using `Normalize((0.5,), (0.5,))` to align with the Generator's output activation (Tanh)
- **DataLoader**: A PyTorch DataLoader is used with a batch size of 32 and shuffling enabled to facilitate efficient batch processing during training.

### 🏗️ Model Architecture

The GAN consists of two neural networks: the **Generator** and the **Discriminator**.

#### 🎨 Generator

The Generator takes a 100-dimensional random noise vector as input and produces a 28x28 image. Its architecture is a fully connected neural network with the following layers:

- **Input**: 100-dimensional noise vector
- **Layers**:
  - Linear (100 → 256) + LeakyReLU (0.2)
  - Linear (256 → 512) + LeakyReLU (0.2)
  - Linear (512 → 1024) + LeakyReLU (0.2)
  - Linear (1024 → 784) + Tanh
- **Output**: Reshaped to (batch_size, 1, 28, 28) to match MNIST image dimensions
- **Activation**: LeakyReLU is used to prevent vanishing gradients, and Tanh ensures the output pixel values are in [-1, 1]

#### 🔍 Discriminator

The Discriminator takes a 28x28 image (flattened to 784 dimensions) and outputs a probability indicating whether the image is real or fake. Its architecture is:

- **Input**: Flattened 784-dimensional image
- **Layers**:
  - Linear (784 → 1024) + LeakyReLU (0.2) + Dropout (0.3)
  - Linear (1024 → 512) + LeakyReLU (0.2) + Dropout (0.3)
  - Linear (512 → 256) + LeakyReLU (0.2) + Dropout (0.3)
  - Linear (256 → 1) + Sigmoid
- **Output**: A single value between 0 and 1, representing the probability of the input being real
- **Dropout**: Applied to prevent overfitting and improve generalization

### 🏋️ Training Process

The GAN is trained for **20 epochs** using the Adam optimizer with a learning rate of 0.0002 for both networks. The training process follows the standard GAN framework:

1. **Loss Function**: Binary Cross-Entropy (BCELoss) is used to compute losses for both the Discriminator and Generator.

2. **Discriminator Training**:
   - Real images are passed through the Discriminator, and the loss is computed against real labels (1s)
   - Fake images generated from random noise are passed through the Discriminator, and the loss is computed against fake labels (0s)
   - The total Discriminator loss is the sum of real and fake losses, and gradients are backpropagated

3. **Generator Training**:
   - The Generator produces fake images from random noise, which are passed through the Discriminator
   - The loss is computed by comparing the Discriminator's output to real labels (1s), encouraging the Generator to produce images that fool the Discriminator

4. **Batch Processing**: Training is performed in mini-batches of 32 images, with the Discriminator and Generator updated alternately in each iteration.

### 📈 Visualization

After training, the Generator produces 32 synthetic images from random noise. These images are visualized in a 4x8 grid using Matplotlib, with each image displayed in grayscale to match the MNIST format.

## 🏆 Achievements

- ✅ **Successful Implementation**: The GAN was successfully implemented using PyTorch, with modular and well-documented code.
- 🎨 **Realistic Image Generation**: The Generator produces synthetic handwritten digits that visually resemble MNIST images after 20 epochs of training.
- ⚖️ **Stable Training**: The use of LeakyReLU, Dropout, and appropriate learning rates ensures stable training without mode collapse.
- 📊 **Visualization**: The project includes clear visualization of generated images, allowing for easy evaluation of the model's performance.
- 📚 **Educational Value**: The code serves as an accessible example for learning about GANs, including data preprocessing, model design, and adversarial training.

## 💻 Installation and Setup

To run this project, ensure you have the following dependencies installed:

- Python 3.6+
- PyTorch
- Torchvision
- Matplotlib
- NumPy

You can install the required packages using pip:

```bash
pip install torch torchvision matplotlib numpy
```

Ensure you have a compatible environment (CPU or GPU). The code automatically downloads the MNIST dataset during execution.

## 🚀 Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/gan-mnist.git
   cd gan-mnist
   ```

2. **Open the Jupyter notebook** `GAN_TC_9926105.ipynb` in a Jupyter environment:
   ```bash
   jupyter notebook GAN_TC_9926105.ipynb
   ```

3. **Run the notebook cells sequentially** to:
   - Load and preprocess the MNIST dataset
   - Define and initialize the Generator and Discriminator
   - Train the GAN for 20 epochs
   - Visualize the generated images

The training process will print the Discriminator and Generator losses for each epoch, and the final cell will display a grid of generated digits.

## 📊 Results

After training for 20 epochs, the Generator produces images that capture the general structure of handwritten digits, such as strokes and shapes. The Discriminator loss stabilizes around 1.0-1.5, and the Generator loss fluctuates between 0.8-1.2, indicating a balanced adversarial training process. The visualized images demonstrate the model's ability to generate recognizable digits, though some may still appear noisy or less refined due to the relatively simple architecture and limited training epochs.

## 🔮 Future Work

- 🏗️ **Improved Architecture**: Incorporate convolutional layers (DCGAN) to enhance the quality of generated images.
- ⚙️ **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and noise dimensions to optimize performance.
- ⏱️ **Extended Training**: Train for more epochs to improve image quality and reduce noise in generated digits.
- 📏 **Evaluation Metrics**: Implement quantitative metrics like Fréchet Inception Distance (FID) to evaluate the quality of generated images.
- 🎯 **Conditional GAN**: Modify the GAN to generate specific digits by conditioning on class labels.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a pull request with a clear description of your changes

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <i>Happy generating! 🎨✨</i>
</div>
