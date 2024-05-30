import model
import utils
import torch
from torch import load
from torch import save
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import nibabel as nib
import numpy as np
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.distributed.hccl
from mpi4py import MPI

__authors__ = 'Conner Swineford and Johanna Walker'
__email__ = 'cswineford@sdsu.edu'
__license__ = 'MIT'

############################################################################
## evaluate.py : This script will run multiple evaluations of the model you
## have trained against the input data specified. This is the script that
## will output the resulting feature map.
############################################################################
## Authors: Conner Swineford and Johanna Walker
## License: MIT License
## Maintainer: Conner Swineford
## Email: cswineford@sdsu.edu
## Status: Production
############################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, help='Path to the data to evaluate the model with')
parser.add_argument('--params', default='./model_weights.pth', type=str, help='Location of the trained model weights')
parser.add_argument('--outdir', default='./', type=str, help='File path to dump script output')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size which must match the batch size used during training')
parser.add_argument('--affine', default='./affine.npy', type=str, help='Path to saved affine numpy array')
args = parser.parse_args()

if __name__ == '__main__':

  device = torch.device('hpu')
  os.environ["PT_HPU_LAZY_MODE"] = "1"

  mpi_comm = MPI.COMM_WORLD
  size = mpi_comm.Get_size()
  rank = mpi_comm.Get_rank()

  if rank == 0:
    print(f'Torch Version: {torch.__version__}')
    print(f'HPU is available: {ht.hpu.is_available()}')
    print(f'Number of devices: {ht.hpu.device_count()}')
    print(f'Device name: {ht.hpu.get_device_name()}')
    print(f'Current Device: {ht.hpu.current_device()}')

  print('INFO, train, size:',size,'rnk:',rank)
  if size >= 1:
    if os.getenv('MASTER_ADDR') is None:
      os.environ['MASTER_ADDR'] = 'localhost'
    if os.getenv('MASTER_POST') is None:
      os.environ['MASTER_POST'] = '12345'

  dist_backend = 'hccl'
  process_per_node = 8
  os.environ['ID'] = str(rank % process_per_node)
  os.environ['LOCAL_RANK'] = str(rank % process_per_node)
  torch.distributed.init_process_group(dist_backend, rank=rank, world_size=size)

  preds, truths = torch.Tensor([]), torch.Tensor([])

  data = utils.import_raw_data(args.indir)
  data = utils.NiiDataset(data['NiiPath'], data['Irr'], data['SubjID'])
  sampler = torch.utils.data.distributed.DistributedSampler(data)
  data = DataLoader(data, batch_size=args.batch_size, num_workers=0, sampler=sampler)
  DIMS = utils.get_loader_dims3d(data)
  n_classes = data.dataset.targets.nunique()

  CNN = model.CNNreg(DIMS).float().to(device)
  PARAMS = load(args.params)
  CNN.load_state_dict(PARAMS, strict=False)

  preds, truths = np.empty((0,2)), np.empty((0,2))
  for X_batch, Y_batch, _ in data:
    Y_batch = Y_batch.to(device)
    X_batch = X_batch.to(device)
    pred = CNN(X_batch)
    pred = pred.detach().cpu().numpy()
    preds = np.concatenate((preds, pred))
    truths = np.concatenate((truths, utils.code_targets(Y_batch).detach().cpu().numpy()))

  gpreds = np.array(mpi_comm.gather(preds, root=0))
  gtruths = np.array(mpi_comm.gather(truths, root=0))

  mpi_comm.barrier()

  if rank == 0:

    gpreds = gpreds.reshape((-1, gpreds.shape[-1]))
    gtruths = gtruths.reshape((-1, gtruths.shape[-1]))

    print('Saving outputs...')  
    save(gpreds, os.path.join(args.outdir, 'predictions.pt'))
    save(gtruths, os.path.join(args.outdir, 'truth.pt'))

  # Retrieve Feature Map:
  m = CNN.HiResRAM(data, rank=rank)

  recvbuf = None
  if rank == 0:
    recvbuf = np.empty((size, 53, 71, 89, 66))

  mpi_comm.Gather(m, recvbuf, root=0)

  mpi_comm.barrier()

  if rank == 0:
    maps = np.array(recvbuf)
    maps = maps.reshape((maps.shape[0]*maps.shape[1],maps.shape[2],maps.shape[3],maps.shape[4]))
    maps = np.mean(maps, axis=0)
    maps = (maps - maps.min()) / (maps.max() - maps.min())
    maps = utils.pad_image(maps, a=10, p=10, t=17, b=8, l=10, r=10)

    mask = nib.load('/home/cswineford/behavioral/MID_mask_95per.nii.gz').dataobj
    maps = np.multiply(maps, mask)

    affine = np.load(args.affine)
    fmap = nib.Nifti1Image(maps, affine)
    print('Saving Feature Map...')
    nib.save(fmap, os.path.join(args.outdir, 'feature_map.nii'))
