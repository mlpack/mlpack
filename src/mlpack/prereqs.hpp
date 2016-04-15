/**
 * @file prereqs.hpp
 *
 * The core includes that mlpack expects; standard C++ includes and Armadillo.
 */
#ifndef MLPACK_PREREQS_HPP
#define MLPACK_PREREQS_HPP

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
#include <iostream>
#include <stdexcept>

// Defining _USE_MATH_DEFINES should set M_PI.
#define _USE_MATH_DEFINES
#include <cmath>

// For tgamma().
#include <boost/math/special_functions/gamma.hpp>

// But if it's not defined, we'll do it.
#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279
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

// We'll need the necessary boost::serialization features, as well as what we
// use with mlpack.  In Boost 1.59 and newer, the BOOST_PFTO code is no longer
// defined, but we still need to define it (as nothing) so that the mlpack
// serialization shim compiles.
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#if BOOST_VERSION < 105500 // Old versions don't have unordered_map support.
  #include "core/boost_backport/unordered_map.hpp"
#else
  #include <boost/serialization/unordered_map.hpp>
#endif
// Boost 1.59 and newer don't use BOOST_PFTO, but our shims do.  We can resolve
// any issue by setting BOOST_PFTO to nothing.
#ifndef BOOST_PFTO
  #define BOOST_PFTO
#endif
#include <mlpack/core/data/serialization_shim.hpp>

// Now include Armadillo through the special mlpack extensions.
#include <mlpack/core/arma_extend/arma_extend.hpp>

// Ensure that the user isn't doing something stupid with their Armadillo
// defines.
#include <mlpack/core/util/arma_config_check.hpp>

// On Visual Studio, disable C4519 (default arguments for function templates)
// since it's by default an error, which doesn't even make any sense because
// it's part of the C++11 standard.
#ifdef _MSC_VER
  #pragma warning(disable : 4519)
#endif


/* From http://gcc.gnu.org/wiki/Visibility */
/* Generic helper definitions for shared library support */
#if defined _WIN32 || defined __CYGWIN__
#define MLPACK_HELPER_DLL_IMPORT __declspec(dllimport)
#define MLPACK_HELPER_DLL_EXPORT __declspec(dllexport)
#define MLPACK_HELPER_DLL_LOCAL
#else
#if __GNUC__ >= 4
#define MLPACK_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
#define MLPACK_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
#define MLPACK_HELPER_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
#else
#define MLPACK_HELPER_DLL_IMPORT
#define MLPACK_HELPER_DLL_EXPORT
#define MLPACK_HELPER_DLL_LOCAL
#endif
#endif

/* Now we use the generic helper definitions above to define MLPACK_API and MLPACK_LOCAL.
 * MLPACK_API is used for the public API symbols. It either DLL imports or DLL exports (or does nothing for static build)
 * MLPACK_LOCAL is used for non-api symbols. */

#ifndef MLPACK_STATIC /* defined if MLPACK is compiled as a DLL */
#ifdef mlpack_EXPORTS /* defined if we are building the MLPACK DLL (instead of using it) */
#define MLPACK_API MLPACK_HELPER_DLL_EXPORT
#else
#define MLPACK_API MLPACK_HELPER_DLL_IMPORT
#endif /* MLPACK_DLL_EXPORTS */
#define MLPACK_LOCAL MLPACK_HELPER_DLL_LOCAL
#else /* MLPACK_STATIC is defined: this means MLPACK is a static lib. */
#define MLPACK_API
#define MLPACK_LOCAL
#endif /* !MLPACK_STATIC */


#endif
