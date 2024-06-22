import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from scipy.ndimage import zoom
import os
import utils

__authors__ = 'Conner Swineford and Johanna Walker'
__email__ = 'cswineford@sdsu.edu'
__license__ = 'MIT'

############################################################################
## model.py : This file contains all the necessary classes and functions for
## the model.
############################################################################
## Authors: Conner Swineford and Johanna Walker
## License: MIT License
############################################################################

class ConvBlock(nn.Module):
  """
  Convolutional block with optional batch normalization and dropout.

  Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
    stride (int, optional): Stride of the convolution. Defaults to 1.
    padding (int, optional): Padding added to all four sides of the input. Defaults to 0.
    norm (bool, optional): Whether to apply batch normalization. Defaults to False.
    dropout (float, optional): Dropout rate. Defaults to 0.
  """
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm=False, dropout=0.):
    super(ConvBlock, self).__init__()
    self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.norm = nn.BatchNorm3d(num_features=out_channels, eps=1e-6)
    self.relu = nn.ReLU()
    self.drop = nn.Dropout3d(p=dropout)

  def forward(self, x):
    """
    Forward pass of the convolutional block.

    Args:
      x (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Output tensor.
    """
    x = self.conv(x)
    x = self.norm(x)
    x = self.relu(x)
    x = self.drop(x)

    return x


class FCBlock(nn.Module):
  """
  Fully connected block with dropout.

  Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    drop (float, optional): Dropout rate. Defaults to 0.2.
  """
  def __init__(self, in_channels, out_channels, drop=0.2):
    super(FCBlock, self).__init__()
    self.drop = nn.Dropout(drop)
    self.fc = nn.Linear(in_channels, out_channels)
    self.in_channels = in_channels

  def forward(self, x):
    """
    Forward pass of the fully connected block.

    Args:
      x (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Output tensor.
    """
    x = self.drop(x)
    x = x.narrow(1, 0, self.in_channels)
    x = self.fc(x)
    return x


class ConvNet(nn.Module):
  """
  Convolutional neural network.

  Args:
    dims (dict): Dictionary containing the dimensions of the input data.
  """
  def __init__(self, dims):
    super(ConvNet, self).__init__()
    
    self.gradients = None
    self.activations = None

    self.conv1 = ConvBlock(1, 8, kernel_size=5, dropout=0.2)
    self.conv2 = ConvBlock(8, 8, kernel_size=5)
    self.conv3 = ConvBlock(8, 16, kernel_size=5, dropout=0.2)
    self.conv4 = ConvBlock(16, 16, kernel_size=5)
    self.conv5 = ConvBlock(16, 32, kernel_size=5, dropout=0.2)
    self.conv6 = ConvBlock(32, 32, kernel_size=5)
    self.conv7 = ConvBlock(32, 64, kernel_size=5, dropout=0.2)

    self.flat = nn.Flatten()
    self.fc1 = FCBlock(147798, 1000)
    self.fc2 = FCBlock(1000, 500)
    self.fc3 = FCBlock(500, 2)

  def gradients_hook(self, module, grad_inp, grad_out):
    """
    Hook to capture gradients during backpropagation.

    Args:
      module (torch.nn.Module): The module to which the hook is attached.
      grad_inp (torch.Tensor): The gradients of the input.
      grad_out (torch.Tensor): The gradients of the output.
    """
    self.gradients = grad_out[0]

  def activations_hook(self, module, args, output):
    """
    Hook to capture activations during forward pass.

    Args:
      module (torch.nn.Module): The module to which the hook is attached.
      args (torch.Tensor): The input arguments.
      output (torch.Tensor): The output tensor.
    """
    self.activations = output

  def forward(self, x):
    """
    Forward pass of the CNN.

    Args:
      x (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Output tensor.
    """
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.conv7(x)

    x = self.flat(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)

    bw = self.conv7.conv.register_full_backward_hook(self.gradients_hook)
    fw = self.conv7.conv.register_forward_hook(self.activations_hook)

    return x

