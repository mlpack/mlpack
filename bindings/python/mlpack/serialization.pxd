#!/usr/bin/python
"""
serialization.pxd: serialization functions for mlpack classes.

This simply makes the utility serialization functions from serialization.hpp
available from Python.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
cimport cython

from libcpp.string cimport string

cdef extern from "serialization.hpp" namespace "mlpack::bindings::python" nogil:
  string SerializeOut[T](T* t, string name) nogil
  void SerializeIn[T](T* t, string str, string name) nogil
  string SerializeOutJSON[T](T* t, string name) nogil
  void SerializeInJSON[T](T* t, string str, string name) nogil
  
