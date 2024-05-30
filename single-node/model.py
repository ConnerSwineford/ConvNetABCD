import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import habana_frameworks.torch.core as htcore
import os
import utils

__authors__ = 'Conner Swineford and Johanna Walker'
__email__ = 'cswineford@sdsu.edu'
__license__ = 'MIT'

############################################################################
## model.py : This file contains all the necessary classes and functions to
## initialize the model.
############################################################################
## Authors: Conner Swineford and Johanna Walker
## License: MIT License
## Maintainer: Conner Swineford
## Email: cswineford@sdsu.edu
## Status: Production
############################################################################


def train_loop(dataloader, model, loss_fn, optimizer, device=torch.device('hpu')):
  '''
  This function performs a single training loop for one epoch of model training.

  Parameters:

  dataloader (torch.utils.data.DataLoader): an iterable PyTorch dataloader object that provides access to the training data.
  model (torch.nn.Module): an initialized PyTorch model to be trained.
  loss_fn (callable): a function that defines the loss for the model.
  optimizer (torch.optim.Optimizer): an optimizer that provides the algorithm for backpropagation.
  
  Returns:

  train_loss (float): the average loss within this epoch.

  Usage:
  Call this function within a training loop to train the model for one epoch. The function takes the dataloader, model, loss_fn,
  and optimizer as input parameters. It then loops over the training data, computes the prediction and loss for each batch,
  performs backpropagation, and updates the model weights. At the end of the epoch, it returns the average loss.

  Note: The tqdm module is used to display a progress bar during training. It is not required for the function to run.
  '''
  train_loss = 0.
  for i, (X, y, _) in enumerate(tqdm(dataloader, position=0, dynamic_ncols=True)):
    # Compute prediction and loss
    pred = model(X.float().to(device))  # Get the prediction output from model
    loss = loss_fn(pred, y.float().unsqueeze(-1).to(device))  # compute loss by calling loss_fn()
    train_loss += int(loss.item())
    # Backpropagation
    optimizer.zero_grad()  # zero_grad()
    loss.backward(retain_graph=True)  # backward()
    htcore.mark_step()
    optimizer.step()  # step()
    htcore.mark_step()

  return train_loss / len(dataloader) # average loss within this epoch


@torch.no_grad()
def val_loop(dataloader, model, loss_fn, device=torch.device('hpu')):
  '''
  This function performs a validation loop to evaluate how well the model generalizes to new data.

  Parameters:

   dataloader (torch.utils.data.DataLoader): an iterable PyTorch dataloader object that provides access to the validation data.
   model (torch.nn.Module): an initialized PyTorch model to be evaluated.
   loss_fn (callable): a function that defines the loss for the model.

  Returns:

   val_loss (float): the average loss of the validation set.
 
  Usage:
   Call this function after each training epoch to evaluate the performance of the model on a separate validation set. The function
   takes the dataloader, model, and loss_fn as input parameters. It then loops over the validation data, computes the prediction
   and loss for each batch, and calculates the average loss. Note that the function is decorated with @torch.no_grad() to ensure
   that gradients are not calculated during validation, which would waste computational resources.

  Note: The validation loop does not update the model weights, as it is only used for evaluation.
  '''

  val_loss = 0.
  for X, y, _ in dataloader:
    # Compute prediction and loss
    pred = model(X.float().to(device)) # Get the prediction output from model
    val_loss += float(loss_fn(pred, y.float().unsqueeze(-1)).item()) # compute loss

  return val_loss / len(dataloader) # average loss of the validation set 


class ConvBlock(nn.Module):
  '''
  This is a PyTorch module that defines a convolutional block used in the neural network. It applies depthwise convolution
  followed by pointwise convolution, group normalization, GELU activation, and GRN (Global Response Normalization) layer. It
  also includes drop path regularization to randomly drop out connections in the residual branch with a certain probability
  during training.

  Parameters:

   in_channels: number of input channels
   drop_path: dropout probability for drop path regularization. If 0 (default), no drop path is applied. If greater than 0,
   a DropPath module is instantiated with the given dropout probability.

  Methods:

  forward(self, x):
   x: input tensor of shape (N, C, H, W, D), where N is the batch size, C is the number of input channels, H is the height,
    W is the width, and D is the depth. The tensor is passed through the convolutional block and returned as a tensor of the
    same shape.
  '''
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm=False, dropout=0.):
    super(ConvBlock, self).__init__()
    self.bnorm = norm
    self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.norm = nn.BatchNorm3d(num_features=out_channels, eps=1e-6)
    self.relu = nn.ReLU()
    self.drop = nn.Dropout3d(p=dropout)

  def forward(self, x):
    x = self.conv(x)
    if self.bnorm:
      x = self.norm(x)
    x = self.relu(x)
    x = self.drop(x)

    return x


class FCBlock(nn.Module):
  def __init__(self, in_channels, out_channels, drop=0.2):
    super(FCBlock, self).__init__()
    self.drop = nn.Dropout(drop)
    self.fc = nn.Linear(in_channels, out_channels)
    self.in_channels = in_channels

  def forward(self, x):
    x = self.drop(x)
    x = x.narrow(1, 0, self.in_channels)
    x = self.fc(x)
    return x


class CNNreg(nn.Module):
  def __init__(self, dims):
    super(CNNreg, self).__init__()
    self.dims = dims
    self.gradients = None
    self.activations = None
    self.device = torch.device('hpu')

    self.conv1 = ConvBlock(1, 4, stride=2)
    self.conv2 = ConvBlock(4, 8, stride=2)
    self.conv3 = ConvBlock(8, 16, kernel_size=5)
    self.conv4 = ConvBlock(16, 32)
    self.conv5 = ConvBlock(32, 64)
    self.conv6 = ConvBlock(64, 128)

    self.flat = nn.Flatten()
    self.fc = FCBlock(294912, 1)

  def gradients_hook(self, module, grad_inp, grad_out):
    self.gradients = grad_out[0]

  def activations_hook(self, module, args, output):
    self.activations = output

  def forward(self, x):

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)

    x = self.flat(x)
    x = self.fc(x)

    bw = self.conv6.conv.register_full_backward_hook(self.gradients_hook)
    fw = self.conv6.conv.register_forward_hook(self.activations_hook)

    return x

  def train(self, loader, epochs=10, loss_fn=nn.MSELoss(), optimizer=Adam, learning_rate=0.0001, graph=True,
            seed=None, outdir='./'):

    train_loss, val_loss = np.zeros(epochs + 1), np.zeros(epochs + 1)

    if type(seed) == int: torch.manual_seed(seed)
    optimizer = optimizer(params=self.parameters(), lr=learning_rate)

    best_loss = float('inf')
    for t in range(epochs):
      print(f'Epoch {t+1}:', end=' ')

      train_loss[t] = train_loop(loader, self, loss_fn, optimizer)
      val_loss[t] = val_loop(loader, self, loss_fn)
      print(f' Training Loss: {train_loss[t]:.5}, Validation Loss: {val_loss[t]:.5}')
      if val_loss[t] < best_loss:
        print(f' Validation loss improved from {best_loss:.5} to {val_loss[t]:.5}. Saving current model...')
        best_loss = val_loss[t]
        torch.save(self.state_dict(), os.path.join(outdir, 'model_weights.pth'))
      else:
        print(f' Validation loss did not improve.')
    self.load_state_dict(torch.load(os.path.join(outdir, 'model_weights.pth')))

    if graph:
      np.save(os.path.join(outdir, 'train_loss.npy'), train_loss)
      np.save(os.path.join(outdir, 'val_loss.npy'), val_loss)

  def eval(self, loader, loss_fn=nn.MSELoss(), get_preds=False):
    preds, Y = np.array([]), np.array([])
    for X_batch, Y_batch, _ in tqdm(loader, position=0, dynamic_ncols=True):
      Y_batch = Y_batch.to(self.device)
      X_batch = X_batch.to(self.device)
      pred = self(X_batch)
      pred = torch.flatten(pred[:,0])
      #pred = torch.flatten(self(X_batch)[:, 0])
      preds = np.concatenate((preds, pred.detach().cpu().numpy()))
      Y = np.concatenate((Y, Y_batch.detach().cpu().numpy()))

    print(float(np.mean(np.absolute(preds-Y))**2))

    #print(f'Evaluation Loss: {float(loss_fn(preds, Y)):>0.2f}')

    if get_preds:
      return Y, preds

  def HiResRAM(self, loader, loss_fn=nn.MSELoss()):
    m = torch.zeros(self.activations[0,0].shape).to(self.device)
    for X_batch, Y_batch, _ in tqdm(loader, position=0, dynamic_ncols=True):
      X_batch = X_batch.to(self.device)
      Y_batch = Y_batch.to(self.device)
      pred = self(X_batch).view(-1)
      loss_fn(pred, Y_batch.float().to(self.device)).backward()
      m = m + torch.multiply((self.activations.detach()*self.gradients.detach()).sum(1), pred[..., None, None,None]).sum(0)
    m = m.detach().cpu().numpy()
    m = utils.resize_volume(m)
    return m
