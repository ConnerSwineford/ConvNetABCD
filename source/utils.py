import torch
import numpy as np
import pandas as pd
from time import localtime
import nibabel as nib
from scipy import ndimage

__authors__ = 'Conner Swineford and Johanna Walker'
__email__ = 'cswineford@sdsu.edu'
__license__ = 'MIT'

############################################################################
## utils.py : This file contains several miscellaneous classes and functions
## for the scripts in this project.
############################################################################
## Author: Conner Swineford and Johanna Walker
## License: MIT License
## Email: cswineford@sdsu.edu
############################################################################


def nii_to_tensor(nifti):
  """
  Converts a NIfTI image to a PyTorch tensor.
    
  Args:
    nifti (nib.Nifti1Image): The NIfTI image to be converted.
        
  Returns:
    torch.Tensor: The image data as a PyTorch tensor.
  """
  return torch.from_numpy(np.expand_dims(np.asarray(nifti.dataobj), axis=0))


def import_raw_data(file_path):
  """
  Imports raw data from a CSV file into a pandas DataFrame.
    
  Args:
    file_path (str): The file path to the CSV file.
        
  Returns:
    pd.DataFrame: The imported data as a pandas DataFrame.
  """
  SubjData = pd.read_csv(file_path, encoding='latin1')
  SubjData = pd.DataFrame(SubjData)
  return SubjData


def get_loader_dims3d(loader):
  """
  Retrieves the dimensions of the data from a data loader.
    
  Args:
    loader (torch.utils.data.DataLoader): The data loader to inspect.
        
  Returns:
    dict: A dictionary containing the batch size, number of channels, 
          height, width, and depth of the data.
  """
  for X, Y, _ in loader:
    dims = {
      'batch_size': X.shape[0],
      'n_channels': X.shape[1],
      'height': X.shape[2],
      'width': X.shape[3],
      'depth': X.shape[4]
    }
    break
  return dims


def get_time_str():
  """
  Gets the current time as a formatted string.
    
  Returns:
    str: The current time in the format 'YYYY_MM_DD_HHMM'.
  """
  return f'{localtime().tm_year}_{localtime().tm_mon:02d}_{localtime().tm_mday:02d}_{localtime().tm_hour:02d}{localtime().tm_min:02d}'


class NiiDataset(torch.utils.data.Dataset):
  """
  A custom Dataset class for loading NIfTI images and their associated labels.
    
  Args:
    paths (list of str): List of file paths to the NIfTI images.
    labels (list of float): List of labels corresponding to the images.
    subjIDs (list of str): List of subject IDs corresponding to the images.
  """
  def __init__(self, paths, labels, subjIDs):
    self.images = [nib.load(image_path) for image_path in paths]
    self.targets = labels
    self.id = subjIDs

  def __len__(self):
    """
    Returns the number of samples in the dataset.
        
    Returns:
      int: The number of samples.
    """
    return len(self.images)

  def __getitem__(self, idx):
    """
    Retrieves a sample and its label from the dataset.
        
    Args:
      idx (int): The index of the sample to retrieve.
            
    Returns:
      tuple: A tuple containing the image tensor, label, and subject ID.
    """
    if type(idx) == int:
      return nii_to_tensor(self.images[idx]), float(self.targets[idx]), self.id[idx]


def compute_accuracy(true_values, predicted_values, alpha=0.1):
  """
  Computes the accuracy of predictions given true values and a tolerance level.
    
  Args:
    true_values (np.ndarray or torch.Tensor): The ground truth values.
    predicted_values (np.ndarray or torch.Tensor): The predicted values.
    alpha (float): The tolerance level for considering a prediction correct.
        
  Returns:
    float: The accuracy of the predictions.
  """
  acc = 0.
  for val in abs(true_values-predicted_values):
    if val > alpha:
      acc += 0.
    else:
      acc += 1.
  return acc / len(true_values)


def resize_volume(img, dims=(91, 109, 91)):
  """
  Resizes a 3D image volume to the specified dimensions.
    
  Args:
    img (np.ndarray): The input 3D image volume.
    dims (tuple of int): The desired dimensions (depth, width, height).
        
  Returns:
    np.ndarray: The resized 3D image volume.
  """
  desired_depth = dims[0]
  desired_width = dims[1]
  desired_height = dims[2]

  current_depth = img.shape[0]
  current_width = img.shape[1]
  current_height = img.shape[2]

  depth = current_depth / desired_depth
  width = current_width / desired_width
  height = current_height / desired_height

  depth_factor = 1 / depth
  width_factor = 1 / width
  height_factor = 1 / height

  img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
  return img

def code_targets(inp:torch.Tensor):
  """
  Converts binary targets into one-hot encoded vectors.
    
  Args:
    inp (torch.Tensor): The input tensor of binary targets.
        
  Returns:
    torch.Tensor: The one-hot encoded target tensor.
  """
  out = torch.empty(inp.shape[0], 2)
  for i in range(len(inp)):
    if float(inp[i]) == 1.:
      out[i] = torch.Tensor([0, 1])
    if float(inp[i]) == 0.:
      out[i] = torch.Tensor([1, 0])
  return out

def resize_image_3d(inp, target=(71, 89, 66)):
  """
  Resizes a 3D image to the specified target dimensions.
    
  Args:
    inp (np.ndarray): The input 3D image.
    target (tuple of int): The target dimensions (depth, width, height).
        
  Returns:
    np.ndarray: The resized 3D image.
  """
  inp_dim = inp.shape
  mult = (1, target[0]/inp_dim[1], target[1]/inp_dim[2], target[2]/inp_dim[3])
  out = ndimage.zoom(inp, zoom=mult)
  return out

def pad_image(inp, a, p, t, b, l, r):
  """
  Pads a 3D image with zeros on specified sides.
    
  Args:
    inp (np.ndarray): The input 3D image.
    a (int): Padding to add to the anterior side.
    p (int): Padding to add to the posterior side.
    t (int): Padding to add to the top side.
    b (int): Padding to add to the bottom side.
    l (int): Padding to add to the left side.
    r (int): Padding to add to the right side.
        
  Returns:
    np.ndarray: The padded 3D image.
  """
  cor,sag,tran = inp.shape
  padded = np.zeros((l + cor + r, p + sag + a, b + tran + t))
  padded[l:cor+l, p:sag+p, b:tran+b] = inp
  return padded

