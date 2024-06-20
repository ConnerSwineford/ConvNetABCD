import os
import model
import utils
import argparse
import time
import pickle
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import nibabel as nib


# Argument parser setup for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to the csv of the training data')
parser.add_argument('--batch_size', default=16, type=int, help='Number of subjects in one model iteration')
parser.add_argument('--epochs', default=10, type=int, help='Number of iterations through the entire data set')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--affine', default='./affine.npy', type=str, help='Path to the saved affine numpy array')
parser.add_argument('--outdir', default='./', type=str, help='File path to dump script output')
parser.add_argument('--workers', type=int, default=1, help='Number of processes for multiprocessing')
parser.add_argument('--weights', default=None, type=str,
                    help='(Optional) If you attempting to initialize the weights of the model, assign the path to the weights here')
parser.add_argument('--seed', default=None, help='(Optional) Random seed for replicability')
args = parser.parse_args()


# Define the training function
def train(rank, world_size):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Set up the process group
  dist.init_process_group("gloo", rank=rank, world_size=world_size)

  print(f"Rank {rank} Initialized")

  # Import and prepare training data
  train = utils.import_raw_data(args.data_path) 
  train_data = utils.NiiDataset(train['NiiPath'], train['Irr'], train['SubjID'])
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
  train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=0, sampler=train_sampler)

  print(f'Data Import Complete [{rank}]')

  # Initialize model and get input dimensions
  dims = utils.get_loader_dims3d(train_loader)
    
  # Create the model
  CNN = model.ConvNet(dims).float().to(device)
  ddp_model = DDP(CNN)
    
  print(f'Model Initialized [{rank}]')
  
  if rank == 0:
    start = time.time()

  torch.distributed.barrier()  

  CNN.train(train_loader, epochs=args.epochs, learning_rate=args.lr, graph=True, seed=args.seed, outdir=args.outdir, rank=rank)

  if rank == 0:
    print('Training Time:', time.time()-start)

  # Clean up
  dist.destroy_process_group()

if __name__ == "__main__":

  # Number of processes to use
  world_size = args.workers
   
  # Set environment variables for master address and port
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
 
  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Use multiprocessing to spawn processes
  mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

