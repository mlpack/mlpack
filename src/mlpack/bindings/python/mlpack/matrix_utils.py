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
# The CategoricalDtype class has moved multiple times, so this insanity is
# necessary to import the right version.
if int(pd.__version__.split('.')[0]) > 0 or \
   int(pd.__version__.split('.')[1]) >= 20:
  from pandas.api.types import CategoricalDtype
elif int(pd.__version__.split('.')[1]) >= 18:
  from pandas.types.dtypes import CategoricalDtype
elif int(pd.__version__.split('.')[1]) == 17:
  from pandas.core.dtypes import CategoricalDtype
elif int(pd.__version__.split('.')[1]) >= 15:
  from pandas.core.common import CategoricalDtype

# We need a unicode type, but on python3 we don't have it.
try:
  UNICODE_EXISTS = bool(type(unicode))
except NameError:
  unicode = str

# We also need a buffer type.
try:
  BUFFER_EXISTS = bool(type(buffer))
except:
  buffer = memoryview

def to_matrix(x, dtype=np.double, copy=False):
  """
  Given some array-like X, return a numpy ndarray of the same type.
  """
  # Make sure it's array-like at all.
  if not hasattr(x, '__len__') and \
      not hasattr(x, 'shape') and \
      not hasattr(x, '__array__'):
    raise TypeError("given argument is not array-like")

  if (isinstance(x, np.ndarray) and x.dtype == dtype and x.flags.c_contiguous):
    if copy: # Copy the matrix if required.
      return x.copy("C"), True
    else:
      return x, False
  elif (isinstance(x, np.ndarray) and x.dtype == dtype and x.flags.f_contiguous):
    # A copy is always necessary here.
    return x.copy("C"), True
  else:
    if isinstance(x, pd.core.series.Series) or isinstance(x, pd.DataFrame):
      # We can only avoid a copy if the dtype is the same and the copy flag is
      # false.  I'm actually not sure if this is possible, since in everything I
      # have found, Pandas stores with F_CONTIGUOUS not C_CONTIGUOUS.
      y = x.values
      if copy == False and y.dtype == dtype and y.flags.c_contiguous:
        return np.ndarray(y.shape, buffer=x.values, dtype=dtype, order='C'),\
            False
      else:
        # We have to make a copy or change the dtype, so just do this directly.
        return np.array(y, dtype=dtype, order='C', copy=True), True
    else:
      return np.array(x, copy=True, dtype=dtype, order='C'), True


def to_matrix_with_info(x, dtype, copy=False):
  """
  Given some array-like X (which should be either a numpy ndarray or a pandas
  DataFrame), convert it into a numpy matrix of the given dtype.
  """
  # Make sure it's array-like at all.
  if not hasattr(x, '__len__') and \
      not hasattr(x, 'shape') and \
      not hasattr(x, '__array__'):
    raise TypeError("given argument is not array-like")

  if isinstance(x, np.ndarray):
    # It is already an ndarray, so the vector of info is all 0s (all numeric).
    if len(x.shape) < 2:
      d = np.zeros(1, dtype=bool)
    else:
      d = np.zeros([x.shape[1]], dtype=bool)

    # Copy the matrix if needed.
    if copy:
      return (x.copy(order="C"), True, d)
    else:
      return (x, False, d)

  if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
    # It's a pandas dataframe.  So we need to see if any of the dtypes are
    # categorical or object, and if so, we need to convert them.  First see if
    # we can take a shortcut without copying.
    dtype_array = x.dtypes.values if len(x.dtypes) > 0 else [x.dtypes]
    if not any(isinstance(t, CategoricalDtype)
        for t in dtype_array) and \
       not np.dtype(object) in dtype_array and \
       not np.dtype(str) in dtype_array and \
       not np.dtype(unicode) in dtype_array:
        # We can just return the matrix as-is; it's all numeric.
        t = to_matrix(x, dtype=dtype, copy=copy)
        if len(x.shape) < 2:
          d = np.zeros(1, dtype=bool)
        else:
          d = np.zeros([x.shape[1]], dtype=bool)
        return (t[0], t[1], d)

    if np.dtype(str) in dtype_array or np.dtype(unicode) in dtype_array:
      raise TypeError('cannot convert matrices with string types')

    if np.dtype(buffer) in dtype_array:
      raise TypeError("'buffer' dtype not supported")

    # If we get to here, then we are going to need to do some type conversion,
    # so go ahead and copy the dataframe and we'll work with y to make
    # modifications.
    y = x
    d = np.zeros([x.shape[1]], dtype=bool)

    # Convert any 'object', 'str', or 'unicode' types to categorical.
    convertColumns = x.select_dtypes(['object'])
    if not convertColumns.empty:
      y[convertColumns] = y[convertColumns].astype('category')

    catColumns = x.select_dtypes(['category']).columns
    if len(catColumns) > 0:
      # Do actual conversion to numeric types.  This converts to an int type.
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

    # We'll have to force the second part of the tuple (whether or not to take
    # ownership) to true.
    t = to_matrix(y.apply(pd.to_numeric), dtype=dtype)
    return (t[0], True, d)

  if isinstance(x, list):
    # Get the number of dimensions.
    dims = 0
    if isinstance(x[0], list):
      dims = len(x[0])
    else:
      dims = len(x)

    d = np.zeros([dims])
    if np.lib.NumpyVersion(np.__version__) >= '2.0.0b1':
      out = np.array(x, dtype=dtype, copy=(True if copy else None))
    else:
      out = np.array(x, dtype=dtype, copy=copy)

    # Since we don't have a great way to check if these are using the same
    # memory location, we will probe manually (ugh).
    oldval = x[0][0]
    x[0][0] *= 2
    alias = False
    if out.flat[0] == x[0][0]:
      alias = True
    x[0][0] = oldval

    return (out, not alias, d)

  # If we got here, the type is not known.
  raise TypeError("given matrix is not a numpy ndarray or pandas DataFrame or "\
      "Python array; not supported at this time");
