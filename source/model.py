import torch
import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm
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

def train_loop(dataloader, model, loss_fn, optimizer, rank=0):
  """
  Training loop for the model.

  Args:
    dataloader (torch.utils.data.DataLoader): The dataloader for the training data.
    model (torch.nn.Module): The model to train.
    loss_fn (torch.nn.Module): The loss function.
    optimizer (torch.optim.Optimizer): The optimizer.
    device (torch.device, optional): The device to use for training. Defaults to HPU.

  Returns:
    float: The average training loss.
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_loss = 0.
  for i, (X, y, _) in enumerate(tqdm(dataloader)):
    # Compute prediction and loss
    optimizer.zero_grad()  # zero_grad()
    pred = model(X.float().to(device)) # Get the prediction output from model
    loss = loss_fn(pred, torch.autograd.Variable(utils.code_targets(y).float().to(device), requires_grad=True))  # compute loss by calling loss_fn()
    # Backpropagation
    loss.backward()  # backward()
    optimizer.step()  # step()
    train_loss += int(loss.item())

  return train_loss / len(dataloader) # average loss within this epoch


@torch.no_grad()
def val_loop(dataloader, model, loss_fn):
  """
  Validation loop for the model.

  Args:
    dataloader (torch.utils.data.DataLoader): The dataloader for the validation data.
    model (torch.nn.Module): The model to validate.
    loss_fn (torch.nn.Module): The loss function.
    device (torch.device, optional): The device to use for validation. Defaults to HPU.

  Returns:
    float: The average validation loss.
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  val_loss = 0.
  for X, y, _ in dataloader:
    # Compute prediction and loss
    pred = model(X.float().to(device)) # Get the prediction output from model
    val_loss += float(loss_fn(pred, utils.code_targets(y).float()).item()) # compute loss

  return val_loss / len(dataloader) # average loss of the validation set 


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
    self.dims = dims
    self.gradients = None
    self.activations = None
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

  def train(self, loader, epochs=10, loss_fn=nn.BCEWithLogitsLoss(), optimizer=Adam, learning_rate=0.00001, graph=True,
            seed=None, outdir='./', rank=0):
    """
    Train the CNN model.

    Args:
      loader (torch.utils.data.DataLoader): DataLoader for the training data.
      epochs (int, optional): Number of epochs to train. Defaults to 10.
      loss_fn (torch.nn.Module, optional): Loss function. Defaults to nn.BCEWithLogitsLoss().
      optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to Adam.
      learning_rate (float, optional): Learning rate. Defaults to 0.00001.
      graph (bool, optional): Whether to plot loss graphs. Defaults to True.
      seed (int, optional): Random seed for reproducibility. Defaults to None.
      outdir (str, optional): Directory to save the model and loss values. Defaults to './'.
      rank (int, optional): Rank of the process in distributed training. Defaults to 0.
    """
    train_loss, val_loss = np.zeros(epochs + 1), np.zeros(epochs + 1)

    if type(seed) == int: torch.manual_seed(seed)
    optimizer = optimizer(params=self.parameters(), lr=learning_rate)

    best_loss = float('inf')
    for t in range(epochs):
      if rank==0:
        print(f'Epoch {t+1}:', end=' ')

      train_loss[t] = train_loop(loader, self, loss_fn, optimizer, rank=rank)
      
      if rank == 0:  
        val_loss[t] = val_loop(loader, self, loss_fn)
        print(f' Training Loss: {train_loss[t]:.5}, Validation Loss: {val_loss[t]:.5}')
        if val_loss[t] < best_loss:
          print(f' Validation loss improved from {best_loss:.5} to {val_loss[t]:.5}. Saving current model...')
          best_loss = val_loss[t]
          torch.save(self.state_dict(), os.path.join(outdir, 'model_weights.pth'), pickle_protocol=4)
        else:
          print(f' Validation loss did not improve.')

      torch.distributed.barrier()

    if rank == 0:
      self.load_state_dict(torch.load(os.path.join(outdir, 'model_weights.pth')))
      if graph:
        np.save(os.path.join(outdir, 'train_loss.npy'), train_loss)
        np.save(os.path.join(outdir, 'val_loss.npy'), val_loss)


  def HiResRAM(self, loader, loss_fn=nn.MSELoss(), rank=0, device='cpu'):
    """
    Computes high-resolution region activation mapping (RAM).
        
    Args:
      loader (torch.utils.data.DataLoader): The data loader.
      loss_fn (callable): The loss function.
      rank (int): Rank of the process for distributed training.
        
    Returns:
      np.ndarray: The RAM values.
    """
    m = np.empty((0, self.dims['height'], self.dims['width'], self.dims['depth']))
    #for X_batch, Y_batch, _ in enumerate(tqdm(loader)):
    for X_batch, Y_batch, _ in tqdm(loader):
      X_batch = X_batch.to(self.device)
      Y_batch = Y_batch.to(self.device)
      pred = self(X_batch)
      loss_fn(pred, utils.code_targets(Y_batch.float()).to(self.device)).backward()
      if self.activations is not None:
          m = np.concatenate((m, utils.resize_image_3d((self.activations.detach().cpu().numpy()*self.gradients.detach().cpu().numpy()).sum(1))), 0)

    return torch.Tensor(m)

