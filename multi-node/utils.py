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
## Authors: Conner Swineford and Johanna Walker
## License: MIT License
## Maintainer: Conner Swineford
## Email: cswineford@sdsu.edu
## Status: Production
############################################################################


def nii_to_tensor(nifti):
  '''
  This function converts a NIfTI image to a PyTorch tensor.

  Parameters:

   nifti: A NIfTI image object
  
  Returns: A PyTorch tensor with shape (1, height, width, depth) where the first dimension is added
           to match the expected input shape of the CNN model.
  '''
  return torch.from_numpy(np.expand_dims(np.asarray(nifti.dataobj), axis=0))


def import_raw_data(file_path):
  """
  Reads in a CSV file from a given file path and returns it as a pandas DataFrame.

  Args:
   file_path (str): The file path of the CSV file to be imported.

  Returns:
   pandas.DataFrame: The imported data as a pandas DataFrame.

  Example:
   >>> import_raw_data('data.csv')
            ID  Age  Gender  ...
        0   1   22   M       ...
        1   2   35   F       ...
        2   3   43   M       ...
        ... ... ...  ...     ...
  """
  SubjData = pd.read_csv(file_path, encoding='latin1')
  SubjData = pd.DataFrame(SubjData)
  return SubjData


def get_loader_dims3d(loader):
  """
  Gets the dimensions of the data in a 3D DataLoader.

  Parameters:
   loader (torch.utils.data.DataLoader): The DataLoader containing the data.

  Returns:
   dict: A dictionary containing the dimensions of the data in the DataLoader,
         with keys 'batch_size', 'n_channels', 'height', 'width', and 'depth'.
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
  '''
  This function returns a string containing the current date and time in the format of "yyyy_mm_dd_hhmm", 
  where yyyy is the year, mm is the month (with leading zero), dd is the day (with leading zero),
  hh is the hour (with leading zero), and mm is the minute (with leading zero).

  Parameters: None

  Returns: A string representing the current date and time in the format of "yyyy_mm_dd_hhmm".
  '''
  return f'{localtime().tm_year}_{localtime().tm_mon:02d}_{localtime().tm_mday:02d}_{localtime().tm_hour:02d}{localtime().tm_min:02d}'


class NiiDataset(torch.utils.data.Dataset):
  """
    A PyTorch Dataset class for loading 3D medical images in NIfTI format.

    Args:
     paths (list): A list of file paths to NIfTI images
     labels (list): A list of corresponding labels for each image
     subjIDs (list): A list of corresponding subject IDs for each image

    Methods:
     __len__(self): Returns the length of the dataset (i.e. number of images)
     __getitem__(self, idx): Returns a tuple containing a torch.Tensor of the image data, the corresponding label as
                             a float, and the subject ID as a string. The image data is loaded from disk using the 
                             nibabel library and converted to a torch.Tensor using the nii_to_tensor() function defined above.
    """
  def __init__(self, paths, labels, subjIDs):
    """
    Initializes a new instance of the NiiDataset class.

    Args:
     paths (list): A list of file paths to NIfTI images
     labels (list): A list of corresponding labels for each image
     subjIDs (list): A list of corresponding subject IDs for each image
    """
    self.images = [nib.load(image_path) for image_path in paths]
    self.targets = labels
    self.id = subjIDs

  def __len__(self):
    """
    Returns the length of the dataset (i.e. number of images).
    """
    return len(self.images)

  def __getitem__(self, idx):
    """
    Returns a tuple containing a torch.Tensor of the image data, the corresponding label as a float, and the subject
    ID as a string. The image data is loaded from disk using the nibabel library and converted to a torch.Tensor
    using the nii_to_tensor() function defined above.

    Args:
     idx (int): The index of the image to retrieve

    Returns:
     tuple: A tuple containing the image data as a torch.Tensor, the corresponding label as a float, and the
            subject ID as a string.
        """
    if type(idx) == int:
      return nii_to_tensor(self.images[idx]), float(self.targets[idx]), self.id[idx]


def compute_accuracy(true_values, predicted_values, alpha=0.1):
  """
  Computes the accuracy of predicted values against true values given a threshold.
    
  Args:
   true_values: A list of true values.
   predicted_values: A list of predicted values.
   alpha: A threshold value, default is 0.1.
    
  Returns:
   A float value representing the accuracy of the predicted values.
  """
  acc = 0.
  for val in abs(true_values-predicted_values):
    if val > alpha:
      acc += 0.
    else:
      acc += 1.
  return acc / len(true_values)

'''class ReLUN(nn.Hardtanh):
    def __init__(self, n_classes, inplace: bool = False):
        super().__init__(0., n_classes, inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str'''

def resize_volume(img, dims=(91, 109, 91)):

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

  out = torch.empty(inp.shape[0], 2)
  for i in range(len(inp)):
    if float(inp[i]) == 1.:
      out[i] = torch.Tensor([0, 1])
    if float(inp[i]) == 0.:
      out[i] = torch.Tensor([1, 0])
  return out

def resize_image_3d(inp, target=(71, 89, 66)):

  inp_dim = inp.shape
  mult = (1, target[0]/inp_dim[1], target[1]/inp_dim[2], target[2]/inp_dim[3])
  out = ndimage.zoom(inp, zoom=mult)
  return out

def pad_image(inp, a, p, t, b, l, r):
  
  cor,sag,tran = inp.shape
  padded = np.zeros((l + cor + r, p + sag + a, b + tran + t))
  padded[l:cor+l, p:sag+p, b:tran+b] = inp
  return padded

