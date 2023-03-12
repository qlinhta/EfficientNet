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
- `SqueezeExcite`: A squeeze-and-excitation block that learns to adaptively recalibrate channel-wise feature responses by
  leveraging global information.
- `InvertedResidualBlock`: An inverted residual block that is optimized for mobile and embedded devices by reducing the
  number of parameters and computation cost.
- `EfficientNet`: The main EfficientNet model that consists of multiple `InvertedResidualBlocks` and a fully connected
  classification head.

The `EfficientNet` class takes two arguments: `version` and `num_classes`. The version argument specifies the version of the
EfficientNet model to use (e.g., `b0`, `b1`, `b2`, etc.), while the `num_classes` argument specifies the number of output
classes for the classification task.

The `calculate_factors` method calculates the width factor, depth factor, and dropout rate for a given `EfficientNet`
version, based on the `phi` value and other hyperparameters defined in the `phi_values` dictionary.

The `create_blocks` method creates the `InvertedResidualBlock` layers for the `EfficientNet` model, based on the
`depth_factor` and `width_factor` values.

The `forward` method performs a forward pass of the input through the `EfficientNet` model.
