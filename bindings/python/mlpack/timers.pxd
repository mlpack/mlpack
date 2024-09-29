#!/usr/bin/env python
"""
timers.pxd: Cython wrapper for Timers.

This file provides a basic wrapper for Timers class, that is used in calling
the main program function.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
cimport cython

cdef extern from "<mlpack/core/util/timers.hpp>" namespace "mlpack::util" nogil:
  cdef cppclass Timers:
    Timers() nogil
