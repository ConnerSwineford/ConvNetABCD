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
import habana_frameworks.torch.core

__authors__ = 'Conner Swineford and Johanna Walker'
__email__ = 'cswineford@sdsu.edu'
__license__ = 'MIT'

############################################################################
## train.py : This is the script that will train train the model to your 
## training data. 
############################################################################
## Authors: Conner Swineford and Johanna Walker
## License: MIT License
## Maintainer: Conner Swineford
## Email: cswineford@sdsu.edu
## Status: Production
############################################################################


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
  print('Torch Version:', torch.__version__)

  device = torch.device('hpu')

  os.environ["PT_HPU_LAZY_MODE"] = "2"

  # Import Data:
  print('Importing Data...')
  train = utils.import_raw_data(args.train_data)
  train_data = utils.NiiDataset(train['NiiPath'], train['Irr'], train['SubjID'])
  train_loader = DataLoader(train_data, batch_size=args.batch_size)

  # Initialize Model:
  print('Initializing Model...')
  dims = utils.get_loader_dims3d(train_loader)
  print(dims)
  
  with open('dims.pkl', 'wb') as f:
    pickle.dump(dims, f)

  n_classes = train_loader.dataset.targets.nunique()
  CNN = model.CNNreg(dims).float().to(device)
  '''if args.weights is not None:
    CNN.load_state_dict(torch.load(args.weights))'''

  # Train Model:
  start = time.time()
  CNN.train(train_loader, epochs=args.epochs, graph=True, seed=args.seed, outdir=args.outdir)
  print('Training Time:', time.time()-start)


