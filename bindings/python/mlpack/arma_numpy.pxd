#!/usr/bin/env python
"""
arma_numpy.pxd: Armadillo/numpy interface functionality.

This file defines a number of functions useful for converting between Armadillo
and numpy objects without actually copying memory.  Because Cython support for
templates is primitive, we can't use templates here and instead overload for
each type we might need.

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

"""
Convert a numpy ndarray to a matrix.
"""
cdef Mat[double]* numpy_to_mat_d(numpy.ndarray[numpy.double_t, ndim=2] X, \
                                 bool takeOwnership) except +
cdef Mat[size_t]* numpy_to_mat_s(numpy.ndarray[numpy.npy_intp, ndim=2] X, \
                                 bool takeOwnership) except +

"""
Convert an Armadillo object to a numpy ndarray of the given type.
"""
cdef numpy.ndarray[numpy.double_t, ndim=2] mat_to_numpy_d(Mat[double]& X) \
    except +
cdef numpy.ndarray[numpy.npy_intp, ndim=2] mat_to_numpy_s(Mat[size_t]& X) \
    except +

"""
Convert a numpy one-dimensional ndarray to a row of the given type.
"""
cdef Row[double]* numpy_to_row_d(numpy.ndarray[numpy.double_t, ndim=1] X, \
                                 bool takeOwnership) except +
cdef Row[size_t]* numpy_to_row_s(numpy.ndarray[numpy.npy_intp, ndim=1] X, \
                                 bool takeOwnership) except +

"""
Convert an Armadillo row vector to a one-dimensional numpy ndarray of the
given type.
"""
cdef numpy.ndarray[numpy.double_t, ndim=1] row_to_numpy_d(Row[double]& X) \
    except +
cdef numpy.ndarray[numpy.npy_intp, ndim=1] row_to_numpy_s(Row[size_t]& X) \
    except +

"""
Convert a numpy one-dimensional ndarray to a column vector of the given type.
"""
cdef Col[double]* numpy_to_col_d(numpy.ndarray[numpy.double_t, ndim=1] X, \
                                 bool takeOwnership) except +
cdef Col[size_t]* numpy_to_col_s(numpy.ndarray[numpy.npy_intp, ndim=1] X, \
                                 bool takeOwnership) except +

"""
Convert an Armadillo column vector to a one-dimensional numpy ndarray of the
given type.
"""
cdef numpy.ndarray[numpy.double_t, ndim=1] col_to_numpy_d(Col[double]& X) \
    except +
cdef numpy.ndarray[numpy.npy_intp, ndim=1] col_to_numpy_s(Col[size_t]& X) \
    except +
