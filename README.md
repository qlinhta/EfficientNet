# EfficientNet

This code implements EfficientNet, a state-of-the-art neural network architecture for image classification. EfficientNet
models are designed to be more accurate and computationally efficient than previous state-of-the-art models, such as
ResNet and Inception.

The paper is available at [arXiv](https://arxiv.org/abs/1905.11946).
Authors: [Mingxing Tan, Quoc V. Le](https://arxiv.org/abs/1905.11946)

The implementation is done using PyTorch.

## Requirements

- Python 3.6 or above
- PyTorch 1.7 or above
- torchvision 0.8 or above

## EfficientNet Architecture

EfficientNet is based on a scalable architecture that uses compound scaling to balance the number of parameters and
computation cost of the network. The architecture consists of several building blocks, including:

- Convolutional Neural Network (CNN) Block
- Squeeze-and-Excitation (SE) Block
- Inverted Residual Block

The EfficientNet architecture is defined by a set of hyperparameters, including the width factor, depth factor, and
dropout rate. These hyperparameters are used to scale the architecture to different levels of computational efficiency
and accuracy.

## Implementation

The implementation includes the following classes:

- `CNNBlock`: A basic convolutional neural network block with batch normalization and SiLU activation.
- `SqueezeExcite`: A squeeze-and-excitation block that learns to adaptively recalibrate channel-wise feature responses
  by
  leveraging global information.
- `InvertedResidualBlock`: An inverted residual block that is optimized for mobile and embedded devices by reducing the
  number of parameters and computation cost.
- `EfficientNet`: The main EfficientNet model that consists of multiple `InvertedResidualBlocks` and a fully connected
  classification head.

The `EfficientNet` class takes two arguments: `version` and `num_classes`. The version argument specifies the version of
the
EfficientNet model to use (e.g., `b0`, `b1`, `b2`, etc.), while the `num_classes` argument specifies the number of
output
classes for the classification task.

The `calculate_factors` method calculates the width factor, depth factor, and dropout rate for a given `EfficientNet`
version, based on the `phi` value and other hyperparameters defined in the `phi_values` dictionary.

The `create_blocks` method creates the `InvertedResidualBlock` layers for the `EfficientNet` model, based on the
`depth_factor` and `width_factor` values.

The `forward` method performs a forward pass of the input through the `EfficientNet` model.

## Usage

To use the `EfficientNet` implementation, simply create an instance of the `EfficientNet` class with the desired version
name and number of classes, and call the `forward` method on the resulting object to perform a forward pass through the
network.

```python
import torch
from efficientnet import EfficientNet

model = EfficientNet(version='b0', num_classes=10)
inputs = torch.randn(1, 3, 224, 224)
outputs = model(inputs)
```

In this example, an instance of the `EfficientNet` class is created with version `b0` and 10 output classes. A random
input
tensor is generated with shape `(1, 3, 224, 224)`, and the forward method is called on the model to generate output
predictions with shape `(1, 10)`.

Note that this implementation of EfficientNet is a simplified version of the architecture described in the original
paper, and does not include all of the advanced techniques and optimizations used in the paper. However, it should be
sufficient for most applications and provides a good starting point for further experimentation and customization.