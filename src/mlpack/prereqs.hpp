/**
 * @file prereqs.hpp
 *
 * The core includes that mlpack expects; standard C++ includes and Armadillo.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_PREREQS_HPP
#define MLPACK_PREREQS_HPP

// Give ourselves a nice way to force functions to be inline if we need.
#define force_inline
#if defined(__GNUG__) && !defined(DEBUG)
  #undef force_inline
  #define force_inline __attribute__((always_inline))
#elif defined(_MSC_VER) && !defined(DEBUG)
  #undef force_inline
  #define force_inline __forceinline
#endif

#include <type_traits>

// Backport this functionality from C++14, if it doesn't exist.
#if __cplusplus <= 201103L
#if !defined(_MSC_VER) || _MSC_VER <= 1800
namespace std {

template<bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;

}
#endif
#endif

// This can be removed with Visual Studio supports an OpenMP version with
// unsigned loop variables.
#ifdef _WIN32
  #define omp_size_t intmax_t
#else
  #define omp_size_t size_t
#endif

// We need to be able to mark functions deprecated.
#include <mlpack/core/util/deprecated.hpp>

#endif
