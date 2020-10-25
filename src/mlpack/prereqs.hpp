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

// Defining _USE_MATH_DEFINES should set M_PI.
#define _USE_MATH_DEFINES
#include <cmath>

// First, check if Armadillo was included before, warning if so.
#ifdef ARMA_INCLUDES
#pragma message "Armadillo was included before mlpack; this can sometimes cause\
 problems.  It should only be necessary to include <mlpack/core.hpp> and not \
<armadillo>."
#endif

// Next, standard includes.
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <climits>
#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>

// But if it's not defined, we'll do it.
#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279
#endif

// MLPACK_COUT_STREAM is used to change the default stream for printing
// purpose.
#if !defined(MLPACK_COUT_STREAM)
 #define MLPACK_COUT_STREAM std::cout
#endif

// MLPACK_CERR_STREAM is used to change the stream for printing warnings
// and errors.
#if !defined(MLPACK_CERR_STREAM)
 #define MLPACK_CERR_STREAM std::cerr
#endif

// Give ourselves a nice way to force functions to be inline if we need.
#define force_inline
#if defined(__GNUG__) && !defined(DEBUG)
  #undef force_inline
  #define force_inline __attribute__((always_inline))
#elif defined(_MSC_VER) && !defined(DEBUG)
  #undef force_inline
  #define force_inline __forceinline
#endif

// Backport this functionality from C++14, if it doesn't exist.
#if __cplusplus <= 201103L
#if !defined(_MSC_VER) || _MSC_VER <= 1800
namespace std {

template<bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;

}
#endif
#endif

// Increase the number of template arguments for the boost list class.
#undef BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#undef BOOST_MPL_LIMIT_LIST_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_LIMIT_LIST_SIZE 50

// We'll need the necessary boost::serialization features, as well as what we
// use with mlpack.  In Boost 1.59 and newer, the BOOST_PFTO code is no longer
// defined, but we still need to define it (as nothing) so that the mlpack
// serialization shim compiles.
#include <boost/serialization/serialization.hpp>
// We are not including boost/serialization/vector.hpp here. It is included in
// mlpack/core/boost_backport/boost_backport_serialization.hpp because of
// different behaviors of vector serialization in different versions of boost.
// #include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
// boost_backport.hpp handles the version and backporting of serialization (and
// other) features.
#include "mlpack/core/boost_backport/boost_backport_serialization.hpp"
// Boost 1.59 and newer don't use BOOST_PFTO, but our shims do.  We can resolve
// any issue by setting BOOST_PFTO to nothing.
#ifndef BOOST_PFTO
  #define BOOST_PFTO
#endif
#include <mlpack/core/data/has_serialize.hpp>
#include <mlpack/core/data/serialization_template_version.hpp>

// If we have Boost 1.58 or older and are using C++14, the compilation is likely
// to fail due to boost::visitor issues.  We will pre-emptively fail.
#if __cplusplus > 201103L && BOOST_VERSION < 105900
#error Use of C++14 mode with Boost < 1.59 is known to cause compilation \
problems.  Instead specify the C++11 standard (-std=c++11 with gcc or clang), \
or upgrade Boost to 1.59 or newer.
#endif

// On Visual Studio, disable C4519 (default arguments for function templates)
// since it's by default an error, which doesn't even make any sense because
// it's part of the C++11 standard.
#ifdef _MSC_VER
  #pragma warning(disable : 4519)
  #define ARMA_USE_CXX11
#endif

// Now include Armadillo through the special mlpack extensions.
#include <mlpack/core/arma_extend/arma_extend.hpp>
#include <mlpack/core/util/arma_traits.hpp>

// Ensure that the user isn't doing something stupid with their Armadillo
// defines.
#include <mlpack/core/util/arma_config_check.hpp>

// All code should have access to logging.
#include <mlpack/core/util/log.hpp>
#include <mlpack/core/util/timers.hpp>

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
