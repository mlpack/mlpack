#!/usr/bin/env python
"""
io.pyx: Cython functionality for mlpack::IO and other utilities.

This file imports the Parameters() function from mlpack::IO, plus other utility
functions: SetParam(), SetParamPtr(), SetParamWithInfo(), GetParam(),
GetParamWithInfo(), EnableVerbose(), DisableVerbose(), DisableBacktrace(),
EnableTimers() and ResetTimers().

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
cimport cython
from libcpp.string cimport string
from libcpp cimport bool
from params cimport Params

cdef extern from "<mlpack/core/util/io.hpp>" namespace "mlpack" nogil:
  cdef cppclass IO:
    @staticmethod
    Params Parameters(string) nogil except +

cdef extern from "<mlpack/bindings/python/mlpack/io_util.hpp>" \
    namespace "mlpack::util" nogil:
  void SetParam[T](Params, string, T&) nogil except +
  void SetParam[T](Params, string, T&, bool) nogil except +
  void SetParamPtr[T](Params, string, T*, bool) nogil except +
  void SetParamWithInfo[T](Params, string, T&, const bool*) nogil except +
  (T*) GetParamPtr[T](Params, string) nogil except +
  (T&) GetParamWithInfo[T](Params, string) nogil except +
  void EnableVerbose() nogil except +
  void DisableVerbose() nogil except +
  void DisableBacktrace() nogil except +
  void ResetTimers() nogil except +
  void EnableTimers() nogil except +
