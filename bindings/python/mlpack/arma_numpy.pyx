#!/usr/bin/env python
"""
arma_numpy.pyx: Armadillo/numpy interface functionality.

This file defines a number of functions useful for converting between Armadillo
and numpy objects without actually copying memory.  Note that if a numpy matrix
is converted to an Armadillo object, then the Armadillo object will "own" the
matrix and free the memory upon destruction (and the numpy object will no longer
"own" the matrix).  Similarly, if an Armadillo object is converted to a numpy
object, then the numpy object will "own" the matrix.

Thus, know that if you convert a matrix type, remember that the resulting type
is what "owns" the allocated memory.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
cimport cython
cimport numpy
import numpy

numpy.import_array()

from .arma cimport Mat, Row, Col
from libcpp cimport bool

import platform
isWin = (platform.system() == "Windows")

cdef extern from "numpy/arrayobject.h":
  void PyArray_ENABLEFLAGS(numpy.ndarray arr, int flags)
  void PyArray_CLEARFLAGS(numpy.ndarray arr, int flags)
  void* PyArray_DATA(numpy.ndarray arr)
  numpy.npy_intp* PyArray_SHAPE(numpy.ndarray arr)
  int PyArray_FLAGS(numpy.ndarray arr)

cdef extern from "<mlpack/bindings/python/mlpack/arma_util.hpp>":
  void SetMemState[T](T& m, int state)
  size_t GetMemState[T](T& m)
  double* GetMemory(Mat[double]& m)
  double* GetMemory(Col[double]& m)
  double* GetMemory(Row[double]& m)
  size_t* GetMemory(Mat[size_t]& m)
  size_t* GetMemory(Col[size_t]& m)
  size_t* GetMemory(Row[size_t]& m)

cdef Mat[double]* numpy_to_mat_d(numpy.ndarray[numpy.double_t, ndim=2] X, \
                                 bool takeOwnership) except +:
  """
  Convert a numpy ndarray to a matrix.  The memory will still be owned by numpy.
  """
  cdef int flags = PyArray_FLAGS(X)
  if not (flags & numpy.NPY_ARRAY_C_CONTIGUOUS) or \
    (not (flags & numpy.NPY_ARRAY_OWNDATA) and not isWin):
    # If needed, make a copy where we own the memory.
    X = X.copy(order="C")
    takeOwnership = True

  cdef Mat[double]* m = new Mat[double](<double*> PyArray_DATA(X),
      PyArray_SHAPE(X)[1], PyArray_SHAPE(X)[0], isWin, False)

  # Take ownership of the memory, if we need to and we are not on Windows.
  if takeOwnership and not isWin:
    PyArray_CLEARFLAGS(X, numpy.NPY_ARRAY_OWNDATA)
    SetMemState[Mat[double]](m[0], 0)

  return m

cdef Mat[size_t]* numpy_to_mat_s(numpy.ndarray[numpy.npy_intp, ndim=2] X, \
                                 bool takeOwnership) except +:
  """
  Convert a numpy ndarray to a matrix.  The memory will still be owned by numpy.
  """
  cdef int flags = PyArray_FLAGS(X)
  if not (flags & numpy.NPY_ARRAY_C_CONTIGUOUS) or \
    (not (flags & numpy.NPY_ARRAY_OWNDATA) and not isWin):
    # If needed, make a copy where we own the memory, except on Windows where
    # we never copy.
    X = X.copy(order="C")
    takeOwnership = True

  cdef Mat[size_t]* m = new Mat[size_t](<size_t*> PyArray_DATA(X),
      PyArray_SHAPE(X)[1], PyArray_SHAPE(X)[0], isWin, False)

  # Take ownership of the memory, if we need to.
  if takeOwnership and not isWin:
    PyArray_CLEARFLAGS(X, numpy.NPY_ARRAY_OWNDATA)
    SetMemState[Mat[size_t]](m[0], 0)

  return m

cdef numpy.ndarray[numpy.double_t, ndim=2] mat_to_numpy_d(Mat[double]& X) \
    except +:
  """
  Convert an Armadillo object to a numpy ndarray.
  """
  # Extract dimensions.
  cdef numpy.npy_intp dims[2]
  dims[0] = <numpy.npy_intp> X.n_cols
  dims[1] = <numpy.npy_intp> X.n_rows
  cdef numpy.ndarray[numpy.double_t, ndim=2] output = \
      numpy.PyArray_SimpleNewFromData(2, &dims[0], numpy.NPY_DOUBLE, GetMemory(X))
  if isWin:
    output = output.copy(order="C")

  # Transfer memory ownership, if needed.
  if GetMemState[Mat[double]](X) == 0 and not isWin:
    SetMemState[Mat[double]](X, 1)
    PyArray_ENABLEFLAGS(output, numpy.NPY_ARRAY_OWNDATA)

  return output

cdef numpy.ndarray[numpy.npy_intp, ndim=2] mat_to_numpy_s(Mat[size_t]& X) \
    except +:
  """
  Convert an Armadillo object to a numpy ndarray.
  """
  # Extract dimensions.
  cdef numpy.npy_intp dims[2]
  dims[0] = <numpy.npy_intp> X.n_cols
  dims[1] = <numpy.npy_intp> X.n_rows
  cdef numpy.ndarray[numpy.npy_intp, ndim=2] output = \
      numpy.PyArray_SimpleNewFromData(2, &dims[0], numpy.NPY_INTP, GetMemory(X))
  if isWin:
    output = output.copy(order="C")

  # Transfer memory ownership, if needed.
  if GetMemState[Mat[size_t]](X) == 0 and not isWin:
    SetMemState[Mat[size_t]](X, 1)
    PyArray_ENABLEFLAGS(output, numpy.NPY_ARRAY_OWNDATA)

  return output

cdef Row[double]* numpy_to_row_d(numpy.ndarray[numpy.double_t, ndim=1] X, \
                                 bool takeOwnership) except +:
  """
  Convert a numpy one-dimensional ndarray to a row.  The memory will still be
  owned by numpy.
  """
  cdef int flags = PyArray_FLAGS(X)
  if not (flags & numpy.NPY_ARRAY_C_CONTIGUOUS) or \
    (not (flags & numpy.NPY_ARRAY_OWNDATA) and not isWin):
    # If needed, make a copy where we own the memory, except on Windows where
    # we never copy.
    X = X.copy(order="C")
    takeOwnership = True

  cdef Row[double]* m = new Row[double](<double*> PyArray_DATA(X),
    PyArray_SHAPE(X)[0], isWin, False)

  # Transfer memory ownership, if needed.
  if takeOwnership and not isWin:
    PyArray_CLEARFLAGS(X, numpy.NPY_ARRAY_OWNDATA)
    SetMemState[Row[double]](m[0], 0)

  return m

cdef Row[size_t]* numpy_to_row_s(numpy.ndarray[numpy.npy_intp, ndim=1] X, \
                                 bool takeOwnership) except +:
  """
  Convert a numpy one-dimensional ndarray to a row.  The memory will still be
  owned by numpy.
  """
  cdef int flags = PyArray_FLAGS(X)
  if not (flags & numpy.NPY_ARRAY_C_CONTIGUOUS) or \
    (not (flags & numpy.NPY_ARRAY_OWNDATA) and not isWin):
    # If needed, make a copy where we own the memory, except on Windows where
    # we never copy.
    X = X.copy(order="C")
    takeOwnership = True

  cdef Row[size_t]* m = new Row[size_t](<size_t*> PyArray_DATA(X),
      PyArray_SHAPE(X)[0], isWin, False)

  # Transfer memory ownership, if needed.
  if takeOwnership and not isWin:
    PyArray_CLEARFLAGS(X, numpy.NPY_ARRAY_OWNDATA)
    SetMemState[Row[size_t]](m[0], 0)

  return m

cdef numpy.ndarray[numpy.double_t, ndim=1] row_to_numpy_d(Row[double]& X) \
    except +:
  """
  Convert an Armadillo row vector to a one-dimensional numpy ndarray.
  """
  # Extract dimensions.
  cdef numpy.npy_intp dim = <numpy.npy_intp> X.n_elem
  cdef numpy.ndarray[numpy.double_t, ndim=1] output = \
      numpy.PyArray_SimpleNewFromData(1, &dim, numpy.NPY_DOUBLE, GetMemory(X))
  if isWin:
    output = output.copy(order="C")

  # Transfer memory ownership, if needed.
  if GetMemState[Row[double]](X) == 0 and not isWin:
    SetMemState[Row[double]](X, 1)
    PyArray_ENABLEFLAGS(output, numpy.NPY_ARRAY_OWNDATA)

  return output

cdef numpy.ndarray[numpy.npy_intp, ndim=1] row_to_numpy_s(Row[size_t]& X) \
    except +:
  """
  Convert an Armadillo row vector to a one-dimensional numpy ndarray.
  """
#  print("called row_to_numpy_s()\n")
  # Extract dimensions.
  cdef numpy.npy_intp dim = <numpy.npy_intp> X.n_elem
  cdef numpy.ndarray[numpy.npy_intp, ndim=1] output = \
      numpy.PyArray_SimpleNewFromData(1, &dim, numpy.NPY_INTP, GetMemory(X))
  if isWin:
    output = output.copy(order="C")

  # Transfer memory ownership, if needed.
  if GetMemState[Row[size_t]](X) == 0 and not isWin:
    SetMemState[Row[size_t]](X, 1)
    PyArray_ENABLEFLAGS(output, numpy.NPY_ARRAY_OWNDATA)

  return output

cdef Col[double]* numpy_to_col_d(numpy.ndarray[numpy.double_t, ndim=1] X, \
                                 bool takeOwnership) except +:
  """
  Convert a numpy one-dimensional ndarray to a column vector.  The memory will
  still be owned by numpy.
  """
  cdef int flags = PyArray_FLAGS(X)
  if not (flags & numpy.NPY_ARRAY_C_CONTIGUOUS) or \
    (not (flags & numpy.NPY_ARRAY_OWNDATA) and not isWin):
    # If needed, make a copy where we own the memory, except on Windows where
    # we never copy.
    X = X.copy(order="C")
    takeOwnership = True

  cdef Col[double]* m = new Col[double](<double*> PyArray_DATA(X),
      PyArray_SHAPE(X)[0], isWin, False)

  # Transfer memory ownership, if needed.
  if takeOwnership and not isWin:
    PyArray_CLEARFLAGS(X, numpy.NPY_ARRAY_OWNDATA)
    SetMemState[Col[double]](m[0], 0)

  return m

cdef Col[size_t]* numpy_to_col_s(numpy.ndarray[numpy.npy_intp, ndim=1] X, \
                                      bool takeOwnership) except +:
  """
  Convert a numpy one-dimensional ndarray to a column vector.  The memory will
  still be owned by numpy.
  """
  cdef int flags = PyArray_FLAGS(X)
  if not (flags & numpy.NPY_ARRAY_C_CONTIGUOUS) or \
    (not (flags & numpy.NPY_ARRAY_OWNDATA) and not isWin):
    # If needed, make a copy where we own the memory, except on Windows where
    # we never copy.
    X = X.copy(order="C")
    takeOwnership = True

  cdef Col[size_t]* m = new Col[size_t](<size_t*> PyArray_DATA(X), 
      PyArray_SHAPE(X)[0], isWin, False)

  # Transfer memory ownership, if needed.
  if takeOwnership and not isWin:
    PyArray_CLEARFLAGS(X, numpy.NPY_ARRAY_OWNDATA)
    SetMemState[Col[size_t]](m[0], 0)

  return m

cdef numpy.ndarray[numpy.double_t, ndim=1] col_to_numpy_d(Col[double]& X) \
    except +:
  """
  Convert an Armadillo column vector to a one-dimensional numpy ndarray.
  """
  # Extract dimension.
  cdef numpy.npy_intp dim = <numpy.npy_intp> X.n_elem
  cdef numpy.ndarray[numpy.double_t, ndim=1] output = \
      numpy.PyArray_SimpleNewFromData(1, &dim, numpy.NPY_DOUBLE, GetMemory(X))
  if isWin:
    output = output.copy(order="C")

  # Transfer memory ownership, if needed.
  if GetMemState[Col[double]](X) == 0 and not isWin:
    SetMemState[Col[double]](X, 1)
    PyArray_ENABLEFLAGS(output, numpy.NPY_ARRAY_OWNDATA)

  return output

cdef numpy.ndarray[numpy.npy_intp, ndim=1] col_to_numpy_s(Col[size_t]& X) \
    except +:
  """
  Convert an Armadillo column vector to a one-dimensional numpy ndarray.
  """
  # Extract dimension.
  cdef numpy.npy_intp dim = <numpy.npy_intp> X.n_elem
  cdef numpy.ndarray[numpy.npy_intp, ndim=1] output = \
      numpy.PyArray_SimpleNewFromData(1, &dim, numpy.NPY_INTP, GetMemory(X))
  if isWin:
    output = output.copy(order="C")

  # Transfer memory ownership, if needed.
  if GetMemState[Col[size_t]](X) == 0 and not isWin:
    SetMemState[Col[size_t]](X, 1)
    PyArray_ENABLEFLAGS(output, numpy.NPY_ARRAY_OWNDATA)

  return output
