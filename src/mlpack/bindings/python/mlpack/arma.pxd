#!/usr/bin/env python
"""
arma.pxd: Armadillo extensions for Cython.

This file defines the Armadillo matrix class and all of the symbols that are
needed for mlpack's Python bindings.  Note that only the necessary symbols are
included here---not the entire Armadillo library.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
cimport cython

from libcpp cimport bool

# We have to include from mlpack/core.hpp so that everything is included in the
# right order and the Armadillo extensions from mlpack are set up right.
cdef extern from "<mlpack/core.hpp>" namespace "arma" nogil:
  # Import the index type.
  ctypedef int uword
  # Import the half-size index type.
  ctypedef int uhword

  cdef cppclass Mat[T]:
    # Special constructor that uses auxiliary memory.
    Mat(T* aux_mem,
        uword n_rows,
        uword n_cols,
        bool copy_aux_mem,
        bool strict) nogil

    # Constructor that initializes memory.
    Mat(uword n_rows, uword n_cols) nogil

    # Default constructor.
    Mat() nogil

    # Number of rows.
    const uword n_rows
    # Number of columns.
    const uword n_cols
    # Total number of elements.
    const uword n_elem

    # Memory state: preallocated, changeable, etc.
    const uhword mem_state

    # Access the memory pointer directly.
    T* memptr() nogil

  cdef cppclass Col[T]:
    # Special constructor that uses auxiliary memory.
    Col(T* aux_mem,
        uword n_elem,
        bool copy_aux_mem,
        bool strict) nogil

    # Constructor that initializes memory.
    Col(uword n_elem) nogil

    # Default constructor.
    Col() nogil

    # Number of rows (equal to number of elements).
    const uword n_rows
    # Number of columns (always 1).
    const uword n_cols
    # Number of elements (equal to number of rows).
    const uword n_elem

    # Memory state: preallocated, changeable, etc.
    const uword mem_state

    # Access the memory pointer directly.
    T* memptr() nogil

  cdef cppclass Row[T]:
    # Special constructor that uses auxiliary memory.
    Row(T* aux_mem,
        uword n_elem,
        bool copy_aux_mem,
        bool strict) nogil

    # Constructor that initializes memory.
    Row(uword n_elem) nogil

    # Default constructor.
    Row() nogil

    # Number of rows (always 1).
    const uword n_rows
    # Number of columns (equal to number of elements).
    const uword n_cols
    # Number of elements (equal to number of columns).
    const uword n_elem

    # Memory state: preallocated, changeable, etc.
    const uword mem_state

    # Access the memory pointer directly.
    T* memptr() nogil
