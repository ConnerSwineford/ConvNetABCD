#!/usr/bin/python

import nibabel as nib
import numpy as np
import sys

affine = np.array(nib.load(sys.argv[1]).affine)
data = np.array(nib.load(sys.argv[1]).dataobj)

data[data < 0.] = 0.
data = data / data.max()

new = nib.Nifti1Image(data, affine)

nib.save(new, str(sys.argv[1]).split('.')[0] + '_norm.nii')

