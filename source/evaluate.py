import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import load, save
from torch.utils.data import DataLoader
from tqdm import tqdm
import nibabel as nib

import model
import utils

# Argument parser setup for command line arguments
parser = argparse.ArgumentParser(description="Evaluate a trained 3D CNN model.")
parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory to evaluate the model with.')
parser.add_argument('--weights', type=str, default='./model_weights.pth', help='Location of the trained model weights.')
parser.add_argument('--workers', type=int, default=1, help='Number of processes for multiprocessing.')
parser.add_argument('--outdir', type=str, default='./', help='File path to save script output.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size; must match the batch size used during training.')
parser.add_argument('--affine', type=str, default='./affine.npy', help='Path to saved affine numpy array.')
args = parser.parse_args()

def hi_res_cam(model, loader, loss_fn=nn.MSELoss(), device='cpu'):
    """
    Computes high-resolution region activation mapping (RAM).

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): The data loader.
        loss_fn (torch.nn.Module): The loss function.
        device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        torch.Tensor: The RAM values as a tensor.
    """
    model = model.module
    dims = utils.get_loader_dims3d(loader)
    activation_maps = np.empty((0, dims['height'], dims['width'], dims['depth']))
    
    for X_batch, Y_batch, _ in tqdm(loader, leave=False, desc='Retreiving Features'):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        pred = model(X_batch)
        loss_fn(pred, utils.code_targets(Y_batch.float()).to(device)).backward()

        if model.activations is not None:
            activations = (model.activations.detach().cpu().numpy() * model.gradients.detach().cpu().numpy()).sum(1)
            resized_activations = utils.resize_image_3d(activations)
            activation_maps = np.concatenate((activation_maps, resized_activations), 0)

    return torch.Tensor(activation_maps)

def load_data(rank, n_workers):
    """
    Load the dataset and create a DataLoader for evaluation.

    Args:
        rank (int): The rank of the current process.
        n_workers (int): Total number of workers for distributed evaluation.

    Returns:
        DataLoader: DataLoader for the evaluation dataset.
    """
    dist.init_process_group("gloo", rank=rank, world_size=n_workers)
    
    data = utils.import_raw_data(args.data_path)
    dataset = utils.NiiDataset(data['NiiPath'], data['Irr'], data['SubjID'])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=n_workers, rank=rank)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, sampler=sampler)
    return loader

def initialize_model(device, dims):
    """
    Initialize the model and load the trained weights.

    Args:
        device (torch.device): The device to run the model on (CPU or GPU).
        dims (dict): Dimensions of the input data.

    Returns:
        torch.nn.Module: The initialized and loaded model.
    """
    model_instance = model.ConvNet(dims).float().to(device)
    params = load(args.weights, map_location=device)
    model_instance.load_state_dict(params, strict=False)
    model_instance = torch.nn.parallel.DistributedDataParallel(model_instance)
    return model_instance
    
def gather_results(preds, truths, rank, n_workers):
    """
    Gather results from all processes and save them.

    Args:
        preds (np.ndarray): Predictions from the current process.
        truths (np.ndarray): Ground truths from the current process.
        rank (int): The rank of the current process.
        n_workers (int): Total number of workers for distributed evaluation.
    """
    gathered_preds = [torch.zeros_like(torch.tensor(preds)) for _ in range(n_workers)] if rank == 0 else None
    gathered_truths = [torch.zeros_like(torch.tensor(truths)) for _ in range(n_workers)] if rank == 0 else None
 
    dist.gather(torch.tensor(preds), gather_list=gathered_preds, dst=0)
    dist.gather(torch.tensor(truths), gather_list=gathered_truths, dst=0)
    
    dist.barrier()
    
    if rank == 0:
        gpreds = np.concatenate([t.numpy() for t in gathered_preds], axis=0)
        gtruths = np.concatenate([t.numpy() for t in gathered_truths], axis=0)
        print('Saving predictions...')
        save(gpreds, os.path.join(args.outdir, 'predictions.pt'))
        save(gtruths, os.path.join(args.outdir, 'truth.pt'))

def feature_attribution(cam, rank, n_workers):
    """
    Gather CAM results from all processes and save them.

    Args:
        cam (torch.Tensor): CAM values from the current process.
        rank (int): The rank of the current process.
        n_workers (int): Total number of workers for distributed evaluation.
    """
    recvbuf = [torch.zeros_like(cam) for _ in range(n_workers)] if rank == 0 else None
    dist.gather(cam, gather_list=recvbuf, dst=0)
    dist.barrier()
    
    if rank == 0:
        maps = np.stack([t.numpy() for t in recvbuf])
        maps = maps.reshape((maps.shape[0] * maps.shape[1], maps.shape[2], maps.shape[3], maps.shape[4]))
        maps = np.mean(maps, axis=0)
        maps = (maps - maps.min()) / (maps.max() - maps.min())
        maps = utils.pad_image(maps, a=10, p=10, t=17, b=8, l=10, r=10)
        
        mask = nib.load('/FolderC/machine_learning/scripts/MID_mask_95per.nii.gz').dataobj
        maps = np.multiply(maps, mask)
        
        affine = np.load(args.affine)
        fmap = nib.Nifti1Image(maps, affine)
        print('Saving Feature Map...')
        nib.save(fmap, os.path.join(args.outdir, 'feature_map.nii'))

def evaluate(rank, n_workers, device):
    """
    Run the evaluation and collect predictions and ground truths.

    Args:
        rank (int): The rank of the current process.
        n_workers (int): Total number of workers for distributed evaluation.
        device (torch.device): The device to run the model on (CPU or GPU).
    """
    loader = load_data(rank, n_workers)
    dims = utils.get_loader_dims3d(loader)
    model_instance = initialize_model(device, dims)

    preds, truths = np.empty((0, 2)), np.empty((0, 2))

    for X_batch, Y_batch, _ in tqdm(loader, leave=False, desc='Retreiving Predictions'):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        pred = model_instance(X_batch).detach().cpu().numpy()
        preds = np.concatenate((preds, pred))
        truths = np.concatenate((truths, utils.code_targets(Y_batch).detach().cpu().numpy()))

    gather_results(preds, truths, rank, n_workers)

    cam = hi_res_cam(model_instance, loader, device=device)

    feature_attribution(cam, rank, n_workers)

    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mp.spawn(evaluate, args=(args.workers, device,), nprocs=args.workers, join=True)

