#!/usr/bin/env python
"""
params.pxd: Cython functionality for mlpack::util::Params.

This file provides the wrapper to the Params class, along with some of
its methods that are:
Get() - Used to "get" a reference to a parameter with the given name.
Has() - Used to know if a parameter with a given name exists in the program.
SetPassed() - Used to set a parameter as "passed".
CheckInputMatrices() - Sanity check for matrics to know if a matrix has NULL
                       or NaN values.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
cimport cython
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "<mlpack/core/util/params.hpp>" namespace "mlpack::util" nogil:
  cdef cppclass Params:
    Params() nogil
    (T&) Get[T](string) nogil except +
    bool Has(string) nogil except +
    void SetPassed(string) nogil except +
    void CheckInputMatrices() nogil except +
