import os
import utils
import argparse

__authors__ = 'Conner Swineford and Johanna Walker'
__email__ = 'cswineford@sdsu.edu'
__license__ = 'MIT'

############################################################################
## main.py : This is the main program for this project. Running this will 
## both train and evaluate your model.
############################################################################
## Authors: Conner Swineford and Johanna Walker
## License: MIT License
## Maintainer: Conner Swineford
## Email: cswineford@sdsu.edu
## Status: Production
############################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, help='Path to the csv of the training data')
parser.add_argument('--test_data', type=str, help='Path to the csv of the data to evaluate the model with')
parser.add_argument('--batch_size', default=64, type=int, help='Number of subjects in one model iteration')
parser.add_argument('--epochs', default=10, type=int, help='Number of iterations through the entire data set')
parser.add_argument('--affine', default='./affine.npy', type=str, help='Path to the saved affine numpy array')
parser.add_argument('--weights', default=None, type=str,
                    help='(Optional) If you attempting to initialize the weights of the model, assign the path to the weights here')
parser.add_argument('--seed', default=None, type=int, help='(Optional) Random seed for replicability')
args = parser.parse_args()


if __name__ == '__main__':
    outdir = os.path.join('/home/cswineford/scripts/outputs/', utils.get_time_str())
    os.makedirs(outdir)

    os.system(f'python scripts/train.py --train_data {args.train_data} --batch_size {args.batch_size} --epochs {args.epochs}'
              f' --affine {args.affine} --outdir {outdir} --affine {args.affine} --weights {args.weights}'
              f' --seed {args.seed}')

    os.system(f"python scripts/evaluate.py --indir {args.test_data} --params {os.path.join(outdir, 'model_weights.pth')}"
              f" --outdir {outdir} --batch_size {args.batch_size}")
