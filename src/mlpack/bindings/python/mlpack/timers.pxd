#!/usr/bin/env python
"""
params.pxd: Cython wrapper for Timers.

This file imports the GetParam() function from mlpack::IO, plus a utility
SetParam() function because Cython can't seem to support lvalue references.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
cimport cython

cdef extern from "<mlpack/core/util/timers.hpp>" namespace "mlpack::util" nogil:
  cdef cppclass Timers:
    Timers() nogil
