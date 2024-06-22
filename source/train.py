import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import save
from tqdm import tqdm
import numpy as np
import nibabel as nib
import time

import model
import utils

# Argument parser setup for command line arguments
parser = argparse.ArgumentParser(description="Train a 3D CNN model.")
parser.add_argument('--data_path', type=str, help='Path to the csv of the training data')
parser.add_argument('--batch_size', default=16, type=int, help='Number of subjects in one model iteration')
parser.add_argument('--epochs', default=10, type=int, help='Number of iterations through the entire data set')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--outdir', default='./', type=str, help='File path to dump script output')
parser.add_argument('--workers', type=int, default=1, help='Number of processes for multiprocessing')
parser.add_argument('--weights', default=None, type=str,
                    help='(Optional) If you attempting to initialize the weights of the model, assign the path to the weights here')
parser.add_argument('--seed', default=None, help='(Optional) Random seed for replicability')
args = parser.parse_args()

def train_loop(dataloader, model, loss_fn, optimizer, epoch, rank=0):
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
    for i, (X, y, _) in enumerate(tqdm(dataloader, leave=False, desc=f'Training Epoch {epoch+1}')):
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

def train(model, loader, epochs, learning_rate, outdir, seed, graph=True, loss_fn=nn.BCEWithLogitsLoss(), optimizer=optim.Adam, rank=0):
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
    optimizer = optimizer(params=model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    for t in range(epochs):
        if rank==0:
            print(f'Epoch {t+1}:', end=' ')

        train_loss[t] = train_loop(loader, model, loss_fn, optimizer, t, rank=rank)

        if rank == 0:
            val_loss[t] = val_loop(loader, model, loss_fn)
            print(f' Training Loss: {train_loss[t]:.5}, Validation Loss: {val_loss[t]:.5}')
            if val_loss[t] < best_loss:
                print(f' Validation loss improved from {best_loss:.5} to {val_loss[t]:.5}. Saving current model...')
                best_loss = val_loss[t]
                torch.save(model.state_dict(), os.path.join(outdir, 'model_weights.pth'), pickle_protocol=4)
            else:
                print(f' Validation loss did not improve.')

        torch.distributed.barrier()

    if rank == 0:
        model.load_state_dict(torch.load(os.path.join(outdir, 'model_weights.pth')))
        if graph:
            np.save(os.path.join(outdir, 'train_loss.npy'), train_loss)
            np.save(os.path.join(outdir, 'val_loss.npy'), val_loss)

# Define the training function
def main(rank, n_workers, device):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up the process group
    dist.init_process_group("gloo", rank=rank, world_size=n_workers)

    # Import and prepare training data
    df = utils.import_raw_data(args.data_path) 
    data = utils.NiiDataset(df['NiiPath'], df['Irr'], df['SubjID'])
    sampler = torch.utils.data.distributed.DistributedSampler(data)
    loader = DataLoader(data, batch_size=args.batch_size, num_workers=0, sampler=sampler)

    # Initialize model and get input dimensions
    dims = utils.get_loader_dims3d(loader)
    
    # Create the model
    CNN = model.ConvNet(dims).float().to(device)
    ddp_model = DDP(CNN)
  
    if rank == 0:
        start = time.time()

    torch.distributed.barrier()  

    train(CNN, loader, epochs=args.epochs, learning_rate=args.lr, outdir=args.outdir, seed=args.seed, graph=True, rank=rank)

    if rank == 0:
        print('Training Time:', time.time()-start)

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
   
    # Set environment variables for master address and port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use multiprocessing to spawn processes
    mp.spawn(main, args=(args.workers, device,), nprocs=args.workers, join=True)

