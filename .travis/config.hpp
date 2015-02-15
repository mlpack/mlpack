// Copyright (C) 2008-2012 NICTA (www.nicta.com.au)
// Copyright (C) 2008-2012 Conrad Sanderson
//
// This file is part of the Armadillo C++ library.
// It is provided without any warranty of fitness
// for any purpose. You can redistribute this file
// and/or modify it under the terms of the GNU
// Lesser General Public License (LGPL) as published
// by the Free Software Foundation, either version 3
// of the License or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)



#if !defined(ARMA_USE_LAPACK)
#define ARMA_USE_LAPACK
//// Uncomment the above line if you have LAPACK or a high-speed replacement for LAPACK,
//// such as Intel's MKL, AMD's ACML, or the Accelerate framework.
//// LAPACK is required for matrix decompositions (eg. SVD) and matrix inverse.
#endif

#if !defined(ARMA_USE_BLAS)
#define ARMA_USE_BLAS
//// Uncomment the above line if you have BLAS or a high-speed replacement for BLAS,
//// such as GotoBLAS, Intel's MKL, AMD's ACML, or the Accelerate framework.
//// BLAS is used for matrix multiplication.
//// Without BLAS, matrix multiplication will still work, but might be slower.
#endif

#define ARMA_USE_WRAPPER
//// Comment out the above line if you prefer to directly link with LAPACK and/or BLAS (eg. -llapack -lblas)
//// instead of linking indirectly with LAPACK and/or BLAS via Armadillo's run-time wrapper library.

// #define ARMA_BLAS_CAPITALS
//// Uncomment the above line if your BLAS and LAPACK libraries have capitalised function names (eg. ACML on 64-bit Windows)

#define ARMA_BLAS_UNDERSCORE
//// Uncomment the above line if your BLAS and LAPACK libraries have function names with a trailing underscore.
//// Conversely, comment it out if the function names don't have a trailing underscore.

// #define ARMA_BLAS_LONG
//// Uncomment the above line if your BLAS and LAPACK libraries use "long" instead of "int"

// #define ARMA_BLAS_LONG_LONG
//// Uncomment the above line if your BLAS and LAPACK libraries use "long long" instead of "int"

// #define ARMA_USE_TBB_ALLOC
//// Uncomment the above line if you want to use Intel TBB scalable_malloc() and scalable_free() instead of standard new[] and delete[]

// #define ARMA_USE_MKL_ALLOC
//// Uncomment the above line if you want to use Intel MKL mkl_malloc() and mkl_free() instead of standard new[] and delete[]

/* #undef ARMA_USE_ATLAS */
#define ARMA_ATLAS_INCLUDE_DIR /
//// If you're using ATLAS and the compiler can't find cblas.h and/or clapack.h
//// uncomment the above define and specify the appropriate include directory.
//// Make sure the directory has a trailing /

#define ARMA_64BIT_WORD
//// Uncomment the above line if you require matrices/vectors capable of holding more than 4 billion elements.
//// Your machine and compiler must have support for 64 bit integers (eg. via "long" or "long long")

#if !defined(ARMA_USE_CXX11)
#define ARMA_USE_CXX11
//// Uncomment the above line if you have a C++ compiler that supports the C++11 standard
//// This will enable additional features, such as use of initialiser lists
#endif

#if !defined(ARMA_USE_HDF5)
/* #undef ARMA_USE_HDF5 */
//// Uncomment the above line if you want the ability to save and load matrices stored in the HDF5 format;
//// the hdf5.h header file must be available on your system and you will need to link with the hdf5 library (eg. -lhdf5)
#endif

#if !defined(ARMA_MAT_PREALLOC)
  #define ARMA_MAT_PREALLOC 16
#endif
//// This is the number of preallocated elements used by matrices and vectors;
//// it must be an integer that is at least 1.
//// If you mainly use lots of very small vectors (eg. <= 4 elements),
//// change the number to the size of your vectors.

#if !defined(ARMA_SPMAT_CHUNKSIZE)
  #define ARMA_SPMAT_CHUNKSIZE 256
#endif
//// This is the minimum increase in the amount of memory (in terms of elements) allocated by a sparse matrix;
//// it must be an integer that is at least 1.
//// The minimum recommended size is 16.

// #define ARMA_NO_DEBUG
//// Uncomment the above line if you want to disable all run-time checks.
//// This will result in faster code, but you first need to make sure that your code runs correctly!
//// We strongly recommend to have the run-time checks enabled during development,
//// as this greatly aids in finding mistakes in your code, and hence speeds up development.
//// We recommend that run-time checks be disabled _only_ for the shipped version of your program.

// #define ARMA_EXTRA_DEBUG
//// Uncomment the above line if you want to see the function traces of how Armadillo evaluates expressions.
//// This is mainly useful for debugging of the library.


// #define ARMA_USE_BOOST
// #define ARMA_USE_BOOST_DATE


#if !defined(ARMA_DEFAULT_OSTREAM)
  #define ARMA_DEFAULT_OSTREAM std::cout
#endif

#define ARMA_PRINT_LOGIC_ERRORS
#define ARMA_PRINT_RUNTIME_ERRORS
//#define ARMA_PRINT_HDF5_ERRORS

#define ARMA_HAVE_STD_ISFINITE
#define ARMA_HAVE_STD_ISINF
#define ARMA_HAVE_STD_ISNAN
#define ARMA_HAVE_STD_SNPRINTF

#define ARMA_HAVE_LOG1P
#define ARMA_HAVE_GETTIMEOFDAY



#if defined(ARMA_DONT_USE_LAPACK)
  #undef ARMA_USE_LAPACK
#endif

#if defined(ARMA_DONT_USE_BLAS)
  #undef ARMA_USE_BLAS
#endif

#if defined(ARMA_DONT_USE_ATLAS)
  #undef ARMA_USE_ATLAS
  #undef ARMA_ATLAS_INCLUDE_DIR
#endif

#if defined(ARMA_DONT_PRINT_LOGIC_ERRORS)
  #undef ARMA_PRINT_LOGIC_ERRORS
#endif

#if defined(ARMA_DONT_PRINT_RUNTIME_ERRORS)
  #undef ARMA_PRINT_RUNTIME_ERRORS
#endif
