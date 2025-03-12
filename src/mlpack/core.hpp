/**
 * @file core.hpp
 *
 * Include all of the base components required to write mlpack methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_HPP
#define MLPACK_CORE_HPP

/**
 * mlpack is an intuitive, fast, and flexible C++ machine learning library with
 * bindings to other languages.  It is meant to be a machine learning analog to
 * LAPACK, and aims to implement a wide array of machine learning methods and
 * function as a "swiss army knife" for machine learning researchers.  The
 * mlpack website can be found at https://mlpack.org.
 *
 * This documentation is API documentation similar to Javadoc.  It isn't
 * necessarily a tutorial, but it does provide detailed documentation on every
 * namespace, method, and class.
 *
 * Each mlpack namespace generally refers to one machine learning method, so
 * browsing the list of namespaces provides some insight as to the breadth of
 * the methods contained in the library.
 *
 * For the list of contributors to mlpack, see
 * https://www.mlpack.org/community.html.  This library would not be possible
 * without everyone's hard work and contributions!
 */

// First, include all of the prerequisites.
#include <mlpack/prereqs.hpp>

// Now the core mlpack classes.
#include <mlpack/core/stb/stb.hpp>
#include <mlpack/core/util/arma_traits.hpp>
#include <mlpack/core/util/ens_traits.hpp>
#include <mlpack/core/util/first_element_is_arma.hpp>
#include <mlpack/core/util/using.hpp>
#include <mlpack/core/util/conv_to.hpp>
#include <mlpack/core/util/distr_param.hpp>
#include <mlpack/core/util/log.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/data/data.hpp>
#include <mlpack/core/math/math.hpp>

// mlpack::backtrace only for linux
#ifdef MLPACK_HAS_BFD_DL
  #include <mlpack/core/util/backtrace.hpp>
#endif

#include <mlpack/core/distances/distances.hpp>
#include <mlpack/core/distributions/distributions.hpp>
#include <mlpack/core/kernels/kernels.hpp>
#include <mlpack/core/metrics/metrics.hpp>
#include <mlpack/core/tree/tree.hpp>

// Include cross-validation and hyperparameter tuning framework.
#include <mlpack/core/cv/cv.hpp>
#include <mlpack/core/hpt/hpt.hpp>

// Use OpenMP if available.
#ifdef MLPACK_USE_OPENMP
  #include <omp.h>
#endif

#endif
