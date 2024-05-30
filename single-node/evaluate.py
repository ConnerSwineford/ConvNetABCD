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
  os.environ["PT_HPU_LAZY_MODE"] = "2"

  data = utils.import_raw_data(args.indir)
  data = utils.NiiDataset(data['NiiPath'], data['Irr'], data['SubjID'])
  data = DataLoader(data, batch_size=args.batch_size)
  DIMS = utils.get_loader_dims3d(data)
  n_classes = data.dataset.targets.nunique()

  CNN = model.CNNreg(DIMS).float().to(device)
  PARAMS = load(args.params)
  CNN.load_state_dict(PARAMS, strict=False)

  true, preds = CNN.eval(data, get_preds=True)

  save(preds, os.path.join(args.outdir, 'predictions.pt'))
  save(true, os.path.join(args.outdir, 'truth.pt'))

  print(f'Average Difference bw Truth and Predicted: {abs(true-preds).mean():.5}')

  acc = []
  for alpha in [0.5, 0.4, 0.3, 0.2, 0.1]:
    acc.append({'Tolerance': alpha, 'Accuracy': utils.compute_accuracy(true, preds, alpha=alpha)})
  accuracies = pd.DataFrame(acc)

  print(accuracies)
  accuracies.to_csv(os.path.join(args.outdir, 'accuracies.csv'))

  fig1 = plt.figure()
  plt.scatter(true, [4] * len(true), label='Actual')
  plt.scatter(preds, [6] * len(preds), label='Predicted')
  plt.vlines(
    x=[true.mean(), preds.mean()],
    ymin=[3.5, 5.5], ymax=[4.5, 6.5],
    color='black', linewidth=2
  )
  plt.vlines(
    x=[true.mean() - true.std(), true.mean() + true.std(),
       preds.mean() - preds.std(), preds.mean() + preds.std()],
    ymin=[3.75, 3.75, 5.75, 5.75], ymax=[4.25, 4.25, 6.25, 6.25],
    color='black', linewidth=2
  )
  plt.title('Boxplot of Predicted Values')
  fig1.legend()
  fig1.savefig(os.path.join(args.outdir, 'boxplot.png'))

  fig2 = plt.figure()
  plt.scatter(true, preds)
  plt.ylabel('Predicted Values')
  plt.xlabel('Actual Values')
  plt.title('Correlation Plot')
  plt.savefig(os.path.join(args.outdir, 'correlation.png'))

  affine = np.load(args.affine)

  # Retrieve Feature Map:
  fmap = nib.Nifti1Image(CNN.HiResRAM(data), affine)
  nib.save(fmap, os.path.join(args.outdir, 'feature_map.nii'))
