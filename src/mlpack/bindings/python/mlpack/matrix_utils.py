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
import pandas as pd

def to_matrix(x):
  """
  Given some array-like X, return a numpy ndarray of the same type.
  """
  # Make sure it's array-like at all.
  if not hasattr(x, '__len__') and \
      not hasattr(x, 'shape') and \
      not hasattr(x, '__array__'):
    raise TypeError("given argument is not array-like")

  return np.array(x, copy=False)

def to_matrix_with_info(x, dtype):
  """
  Given some array-like X (which should be either a numpy ndarray or a pandas
  DataFrame, convert into a numpy matrix of the given dtype.
  """
  # Make sure it's array-like at all.
  if not hasattr(x, '__len__') and \
      not hasattr(x, 'shape') and \
      not hasattr(x, '__array__'):
    raise TypeError("given argument is not array-like")

  if isinstance(x, np.ndarray):
    # It is already an ndarray, so the vector of info is all 0s (all numeric).
    d = np.zeros([x.shape[1]])
    return (x, d)

  if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
    # It's a pandas dataframe.  So we need to see if any of the dtypes are
    # categorical or object, and if so, we need to convert them.  First see if
    # we can take a shortcut without copying.
    if not 'category' in x.dtypes.values and \
       not 'object' in x.dtypes.values and \
       not 'str' in x.dtypes.values and \
       not 'unicode' in x.dtypes.values:
        # We can just return the matrix as-is; it's all numeric.
        d = np.zeros([x.shape[1], 1])
        return (to_matrix(x), d)

    if 'str' in x.dtypes.values or 'unicode' in x.dtypes.values:
      raise TypeError('cannot convert matrices with string types')

    if 'buffer' in x.dtypes.values:
      raise TypeError("'buffer' dtype not supported")

    # If we get to here, then we are going to need to do some type conversion,
    # so go ahead and copy the dataframe and we'll work with y to make
    # modifications.
    y = x
    d = np.zeros([x.shape[1]])

    # Convert any 'object', 'str', or 'unicode' types to categorical.
    convertColumns = x.select_dtypes(['object'])
    if not convertColumns.empty:
      print("cols: ", convertColumns)
      y[convertColumns] = y[convertColumns].astype('category')

    if 'category' in x.dtypes.values:
      # Do actual conversion to numeric types.  This converts to an int type.
      catColumns = x.select_dtypes(['category']).columns
      y = x # Copy it... not great...

      # Note that this will map NaNs (missing values or unknown categories) to
      # -1, so we will have to convert those back to NaN.
      y[catColumns] = y[catColumns].apply(
          lambda c: c.cat.codes).astype('double')
      y[catColumns].replace(to_replace=[-1], value=float('NaN'))

      # Construct dataset information: 1s represent categorical data, 0s
      # represent otherwise.
      catColumnIndices = [y.columns.get_loc(i) for i in catColumns]
      d[catColumnIndices] = 1

    return (to_matrix(y.apply(pd.to_numeric)), d)

  if isinstance(x, list):
    # Get the number of dimensions.
    dims = 0
    if isinstance(x[0], list):
      dims = len(x[0])
    else:
      dims = len(x)

    d = np.zeros([dims])
    return (np.array(x, dtype=dtype), d)

  # If we got here, the type is not known.
  raise TypeError("given matrix is not a numpy ndarray or pandas DataFrame or "\
      "Python array; not supported at this time");
