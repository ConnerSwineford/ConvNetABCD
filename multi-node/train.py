import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import model
import utils
import argparse
import nibabel as nib
import os
import time
import pickle
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.distributed.hccl
from mpi4py import MPI

__authors__ = 'Conner Swineford and Johanna Walker'
__email__ = 'cswineford@sdsu.edu'
__license__ = 'MIT'

############################################################################
## train.py : This is the script that will train train the model to your 
## training data. 
############################################################################
## Authors: Conner Swineford and Johanna Walker
## License: MIT License
## Email: cswineford@sdsu.edu
############################################################################


# Argument parser setup for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, help='Path to the csv of the training data')
parser.add_argument('--batch_size', default=64, type=int, help='Number of subjects in one model iteration')
parser.add_argument('--epochs', default=10, type=int, help='Number of iterations through the entire data set')
parser.add_argument('--affine', default='./affine.npy', type=str, help='Path to the saved affine numpy array')
parser.add_argument('--outdir', default='./', type=str, help='File path to dump script output')
parser.add_argument('--weights', default=None, type=str,
                    help='(Optional) If you attempting to initialize the weights of the model, assign the path to the weights here')
parser.add_argument('--seed', default=None, help='(Optional) Random seed for replicability')
args = parser.parse_args()


if __name__ == '__main__':

  # Set the device to HPU (Habana Processing Unit)
  device = torch.device('hpu')

  # Set the HPU lazy mode for performance optimization
  os.environ["PT_HPU_LAZY_MODE"] = "1"

  # Initialize MPI for distributed training
  mpi_comm = MPI.COMM_WORLD
  size = mpi_comm.Get_size()
  rank = mpi_comm.Get_rank()
  
  # Print out HPU details for the main process (rank 0)
  if rank == 0:
    print(f'Torch Version: {torch.__version__}')
    print(f'HPU is available: {ht.hpu.is_available()}')
    print(f'Number of devices: {ht.hpu.device_count()}')
    print(f'Device name: {ht.hpu.get_device_name()}')
    print(f'Current Device: {ht.hpu.current_device()}')

  print('INFO, train, size:',size,'rnk:',rank)
  
  # Set up the master address and port for distributed training
  if size >= 1:
    if os.getenv('MASTER_ADDR') is None:
      os.environ['MASTER_ADDR'] = 'localhost'
    if os.getenv('MASTER_POST') is None:
      os.environ['MASTER_POST'] = '12345'

  # Initialize the process group for distributed training with Habana HCCL backend
  dist_backend = 'hccl'
  process_per_node = 8
  os.environ['ID'] = str(rank % process_per_node)
  os.environ['LOCAL_RANK'] = str(rank % process_per_node)
  torch.distributed.init_process_group(dist_backend, rank=rank, world_size=size)

  # Import and prepare training data
  if rank == 0:
    print('Importing Data...')
  train = utils.import_raw_data(args.train_data)
  train_data = utils.NiiDataset(train['NiiPath'], train['Irr'], train['SubjID'])
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
  train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=0, sampler=train_sampler)

  # Initialize model and get input dimensions
  dims = utils.get_loader_dims3d(train_loader)
  
  if rank == 0:
    print(dims)
    with open('dims.pkl', 'wb') as f:
      pickle.dump(dims, f)
    print('Initializing Model...')

  # Instantiate the CNN model and move it to the HPU device
  CNN = model.CNNreg(dims).float().to(device)
  
  # Optionally load pre-trained weights
  if args.weights is not None:
    CNN.load_state_dict(torch.load(args.weights))

  if rank == 0:
    print('Distributing Model...')

  # Wrap the model for distributed training
  CNN = torch.nn.parallel.DistributedDataParallel(CNN,
          bucket_cap_mb=100,
          broadcast_buffers=False,
          gradient_as_bucket_view=True)

  # Train the model:
  if rank == 0:
    print('Training Model...')
    start = time.time()

  CNN.module.train(train_loader, epochs=args.epochs, graph=True, seed=args.seed, outdir=args.outdir, rank=rank)
  
  if rank == 0:
    print('Training Time:', time.time()-start)


