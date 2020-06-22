#!/usr/bin/env python
"""
cli.pyx: Cython functionality for mlpack::CLI.

This file imports the GetParam() function from mlpack::CLI, plus a utility
SetParam() function because Cython can't seem to support lvalue references.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
cimport cython

from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "<mlpack/core/util/cli.hpp>" namespace "mlpack" nogil:
  cdef cppclass CLI:
    @staticmethod
    (T&) GetParam[T](string) nogil except +

    @staticmethod
    bool HasParam(string) nogil except +

    @staticmethod
    void SetPassed(string) nogil except +

    @staticmethod
    void Destroy() nogil except +

    @staticmethod
    void StoreSettings(string) nogil except +

    @staticmethod
    void RestoreSettings(string) nogil except +

    @staticmethod
    void ClearSettings() nogil except +

cdef extern from "<mlpack/bindings/python/mlpack/cli_util.hpp>" \
    namespace "mlpack::util" nogil:
  void SetParam[T](string, T&) nogil except +
  void SetParamPtr[T](string, T*, bool) nogil except +
  void SetParamWithInfo[T](string, T&, const bool*) nogil except +
  (T*) GetParamPtr[T](string) nogil except +
  (T&) GetParamWithInfo[T](string) nogil except +
  void EnableVerbose() nogil except +
  void DisableVerbose() nogil except +
  void DisableBacktrace() nogil except +
  void ResetTimers() nogil except +
  void EnableTimers() nogil except +
