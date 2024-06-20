import model
import utils
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import load, save
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import nibabel as nib
import numpy as np

# Argument parser setup for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, help='Path to the data to evaluate the model with')
parser.add_argument('--params', default='./model_weights.pth', type=str, help='Location of the trained model weights')
parser.add_argument('--workers', type=int, default=1, help='Number of processes for multiprocessing')
parser.add_argument('--outdir', default='./', type=str, help='File path to dump script output')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size which must match the batch size used during training')
parser.add_argument('--affine', default='./affine.npy', type=str, help='Path to saved affine numpy array')
args = parser.parse_args()

def evaluate(rank, world_size):
    # Set up the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Set the device to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize tensors for predictions and ground truths
    preds, truths = torch.Tensor([]), torch.Tensor([])

    # Load and prepare evaluation data
    data = utils.import_raw_data(args.indir)
    data = utils.NiiDataset(data['NiiPath'], data['Irr'], data['SubjID'])
    sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=world_size, rank=rank)
    data = DataLoader(data, batch_size=args.batch_size, num_workers=0, sampler=sampler)
    DIMS = utils.get_loader_dims3d(data)
    n_classes = data.dataset.targets.nunique()

    # Initialize model and load trained parameters
    CNN = model.ConvNet(DIMS).float().to(device)
    PARAMS = load(args.params, map_location=device)
    CNN.load_state_dict(PARAMS, strict=False)
    CNN = torch.nn.parallel.DistributedDataParallel(CNN)

    # Prepare for evaluation
    preds, truths = np.empty((0,2)), np.empty((0,2))
    for X_batch, Y_batch, _ in data:
        Y_batch = Y_batch.to(device)
        X_batch = X_batch.to(device)
        pred = CNN(X_batch)
        pred = pred.detach().cpu().numpy()
        preds = np.concatenate((preds, pred))
        truths = np.concatenate((truths, utils.code_targets(Y_batch).detach().cpu().numpy()))

    # Gather predictions and ground truths from all processes
    gathered_preds = [torch.zeros_like(torch.tensor(preds)) for _ in range(world_size)] if rank == 0 else None
    gathered_truths = [torch.zeros_like(torch.tensor(truths)) for _ in range(world_size)] if rank == 0 else None
    dist.gather(torch.tensor(preds), gather_list=gathered_preds, dst=0)
    dist.gather(torch.tensor(truths), gather_list=gathered_truths, dst=0)

    dist.barrier()

    if rank == 0:
        # Convert gathered lists to numpy arrays
        gpreds = np.concatenate([t.numpy() for t in gathered_preds], axis=0)
        gtruths = np.concatenate([t.numpy() for t in gathered_truths], axis=0)

        # Save predictions and ground truths to output directory
        print('Saving outputs...')  
        save(gpreds, os.path.join(args.outdir, 'predictions.pt'))
        save(gtruths, os.path.join(args.outdir, 'truth.pt'))

    # Retrieve feature map using HiResRAM method from model
    m = CNN.module.HiResRAM(data, rank=rank, device=device)

    # Allocate buffer for gathering feature maps on the root process
    recvbuf = None
    if rank == 0:
        recvbuf = [torch.zeros_like(m) for _ in range(args.workers)]

    dist.gather(torch.Tensor(m), gather_list=recvbuf, dst=0)

    dist.barrier()

    if rank == 0:
        
        # Process and save feature maps
        maps = np.stack([t.numpy() for t in recvbuf])
        maps =  maps.reshape((maps.shape[0]*maps.shape[1],maps.shape[2],maps.shape[3],maps.shape[4]))
        maps = np.mean(maps, axis=0)
        maps = (maps - maps.min()) / (maps.max() - maps.min())
        maps = utils.pad_image(maps, a=10, p=10, t=17, b=8, l=10, r=10)

        # Apply brain mask to feature map
        mask = nib.load('/FolderC/machine_learning/scripts/MID_mask_95per.nii.gz').dataobj
        maps = np.multiply(maps, mask)

        # Save the final feature map as a NIfTI image
        affine = np.load(args.affine)
        fmap = nib.Nifti1Image(maps, affine)
        print('Saving Feature Map...')
        nib.save(fmap, os.path.join(args.outdir, 'feature_map.nii'))

    dist.destroy_process_group()

if __name__ == "__main__":
    # Number of processes to use
    world_size = args.workers 
    
    # Set environment variables for master address and port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use multiprocessing to spawn processes
    mp.spawn(evaluate, args=(world_size,), nprocs=world_size, join=True)

