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
#include <tuple>
#include <queue>

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
// boost_backport.hpp handles the version and backporting of serialization (and
// other) features.
#include "mlpack/core/boost_backport/boost_backport.hpp"
// Boost 1.59 and newer don't use BOOST_PFTO, but our shims do.  We can resolve
// any issue by setting BOOST_PFTO to nothing.
#ifndef BOOST_PFTO
  #define BOOST_PFTO
#endif
#include <mlpack/core/data/serialization_shim.hpp>
#include <mlpack/core/data/serialization_template_version.hpp>

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
  #define ARMA_USE_CXX11
#endif

#endif
