#!/usr/bin/env python
"""
matrix_utils.py: utilities for matrix conversion

This file defines the to_matrix() function, which can be used to convert Pandas
dataframes or other types of array-like objects to numpy ndarrays for use in
mlpack bindings.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
import numpy as np

def to_matrix(x):
  """
  Given some array-like X, return an Armadillo matrix.
  """
  # Make sure it's array-like at all.
  if not hasattr(x, '__len__') and \
      not hasattr(x, 'shape') and \
      not hasattr(x, '__array__'):
    raise TypeError("given argument is not array-like")

  return np.array(x, copy=False)
