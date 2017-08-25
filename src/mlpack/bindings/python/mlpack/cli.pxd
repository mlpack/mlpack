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
    (T&) GetParam[T](string) nogil

    @staticmethod
    void SetPassed(string) nogil

    @staticmethod
    void Destroy() nogil

    @staticmethod
    void StoreSettings(string) nogil

    @staticmethod
    void RestoreSettings(string) nogil

    @staticmethod
    void ClearSettings() nogil

cdef extern from "<mlpack/bindings/python/mlpack/cli_util.hpp>" \
    namespace "mlpack::util" nogil:
  void SetParam[T](string, const T&) nogil
  void SetParamWithInfo[T](string, const T&, const bool*) nogil
  (T&) GetParamWithInfo[T](string) nogil
  void EnableVerbose() nogil

cdef extern from "<mlpack/bindings/python/mlpack/move.hpp>" \
    namespace "mlpack::util" nogil:
  void MoveFromPtr[T](T&, T*) nogil
  void MoveToPtr[T](T*, T&) nogil
