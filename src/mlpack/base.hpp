/**
 * @file base.hpp
 *
 * The most basic core includes that mlpack expects; standard C++ includes and
 * Armadillo only
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BASE_HPP
#define MLPACK_BASE_HPP

// Defining _USE_MATH_DEFINES should set M_PI.
#define _USE_MATH_DEFINES
#include <cmath>

// Next, standard includes.
#include <any>
#include <cctype>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <numeric>
#include <vector>
#include <queue>

// But if it's not defined, we'll do it.
#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279
#endif

// Include mlpack configuration.
#ifdef MLPACK_CUSTOM_CONFIG_FILE
  // During the build process, CMake will generate a custom configuration file
  // specific to the system.  When CMake is used to build mlpack, that custom
  // file is used via the MLPACK_CUSTOM_CONFIG_FILE macro.  When mlpack is
  // installed by CMake, the custom configuration file is installed as
  // config.hpp, and MLPACK_CUSTOM_CONFIG_FILE is not set after installation.
  #undef MLPACK_WRAP_INCLUDE
  #define MLPACK_WRAP_INCLUDE(x) <x>
  #include MLPACK_WRAP_INCLUDE(MLPACK_CUSTOM_CONFIG_FILE)
#else
  #include "config.hpp"
#endif

// Give ourselves a nice way to force functions to be inline if we need.
#undef mlpack_force_inline
#define mlpack_force_inline
#if defined(__GNUG__) && !defined(DEBUG)
  #undef mlpack_force_inline
  #define mlpack_force_inline __attribute__((always_inline))
#elif defined(_MSC_VER) && !defined(DEBUG)
  #undef mlpack_force_inline
  #define mlpack_force_inline __forceinline
#endif

// detect C++17 mode
#if (__cplusplus >= 201703L)
  #undef  MLPACK_HAVE_CXX17
  #define MLPACK_HAVE_CXX17
#endif

#if defined(_MSVC_LANG)
  #if (_MSVC_LANG >= 201703L)
    #undef  MLPACK_HAVE_CXX17
    #define MLPACK_HAVE_CXX17
  #endif
#endif

#if !defined(MLPACK_HAVE_CXX17)
  #error "Need to enable C++17 mode in your compiler"
#endif

// Now include Armadillo and traits that we use for it.
#include <armadillo>
#include <mlpack/core/util/arma_traits.hpp>
#include <mlpack/core/util/omp_reductions.hpp>

// On Visual Studio, disable C4519 (default arguments for function templates)
// since it's by default an error, which doesn't even make any sense because
// it's part of the C++11 standard.
#ifdef _MSC_VER
  #pragma warning(disable : 4519)
#endif

// OpenMP usage must be version 3.1 or newer, if it is being used.
#if (defined(_OPENMP) && (_OPENMP >= 201107))
  #undef MLPACK_USE_OPENMP
  #define MLPACK_USE_OPENMP
#elif defined(_OPENMP)
  #ifdef _MSC_VER
    #error "mlpack requires OpenMP 3.1+; compile without /OPENMP"
  #else
    #error "mlpack requires OpenMP 3.1+; disable OpenMP in your compiler"
  #endif
#endif

#endif
