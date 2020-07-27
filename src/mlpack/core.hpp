/**
 * @file core.hpp
 *
 * Include all of the base components required to write mlpack methods, and the
 * main mlpack Doxygen documentation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_HPP
#define MLPACK_CORE_HPP

/**
 * @mainpage mlpack Documentation
 *
 * @section intro_sec Introduction
 *
 * mlpack is an intuitive, fast, and flexible C++ machine learning library with
 * bindings to other languages.  It is meant to be a machine learning analog to
 * LAPACK, and aims to implement a wide array of machine learning methods and
 * function as a "swiss army knife" for machine learning researchers.  The
 * mlpack website can be found at https://mlpack.org.
 *
 * @section howto How To Use This Documentation
 *
 * This documentation is API documentation similar to Javadoc.  It isn't
 * necessarily a tutorial, but it does provide detailed documentation on every
 * namespace, method, and class.
 *
 * Each mlpack namespace generally refers to one machine learning method, so
 * browsing the list of namespaces provides some insight as to the breadth of
 * the methods contained in the library.
 *
 * To generate this documentation in your own local copy of mlpack, you can use
 * the 'doc' CMake target, which is available if CMake has found Doxygen, from
 * the build directory:
 *
 * @code
 * $ make doc
 * @endcode
 *
 * @section tutorial Tutorials
 *
 * A few short tutorials on how to use mlpack are given below.
 *
 *  - @ref build
 *  - @ref build_windows
 *  - @ref matrices
 *  - @ref iodoc
 *  - @ref timer
 *  - @ref sample
 *  - @ref sample_ml_app
 *  - @ref cv
 *  - @ref hpt_guide
 *  - @ref verinfo
 *
 * @section remarks Final Remarks
 *
 * For the list of contributors to mlpack, see
 * https://www.mlpack.org/community.html.  This library would not be possible
 * without everyone's hard work and contributions!
 */

// First, include all of the prerequisites.
#include <mlpack/prereqs.hpp>

// Now the core mlpack classes.
#include <mlpack/core/util/arma_traits.hpp>
#include <mlpack/core/util/log.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/deprecated.hpp>
#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>
#include <mlpack/core/data/normalize_labels.hpp>
#include <mlpack/core/math/clamp.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/math/random_basis.hpp>
#include <mlpack/core/math/lin_alg.hpp>
#include <mlpack/core/math/range.hpp>
#include <mlpack/core/math/round.hpp>
#include <mlpack/core/math/shuffle_data.hpp>
#include <mlpack/core/math/ccov.hpp>
#include <mlpack/core/math/make_alias.hpp>
#include <mlpack/core/dists/discrete_distribution.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>
#include <mlpack/core/dists/laplace_distribution.hpp>
#include <mlpack/core/dists/gamma_distribution.hpp>
#include <mlpack/core/dists/diagonal_gaussian_distribution.hpp>
#include <mlpack/core/data/confusion_matrix.hpp>

// mlpack::backtrace only for linux
#ifdef HAS_BFD_DL
  #include <mlpack/core/util/backtrace.hpp>
#endif

// Include kernel traits.
#include <mlpack/core/kernels/kernel_traits.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/polynomial_kernel.hpp>
#include <mlpack/core/kernels/cosine_distance.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/epanechnikov_kernel.hpp>
#include <mlpack/core/kernels/hyperbolic_tangent_kernel.hpp>
#include <mlpack/core/kernels/laplacian_kernel.hpp>
#include <mlpack/core/kernels/pspectrum_string_kernel.hpp>
#include <mlpack/core/kernels/spherical_kernel.hpp>
#include <mlpack/core/kernels/triangular_kernel.hpp>
#include <mlpack/core/kernels/cauchy_kernel.hpp>

// Use OpenMP if compiled with -DHAS_OPENMP.
#ifdef HAS_OPENMP
  #include <omp.h>
#endif

// Use Armadillo's C++ version detection.
#ifdef ARMA_USE_CXX11
  #define MLPACK_USE_CX11
#endif

#endif
