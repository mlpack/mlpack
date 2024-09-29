/**
 * @file src/rcpp_mlpack.h
 * @author Dirk Eddelbuettel
 * @author Yashwant Singh Parihar
 *
 * Include all of the base components required to work mlpack bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_RCPP_MLPACK_H
#define MLPACK_BINDINGS_R_RCPP_MLPACK_H

#include <Rcpp.h>

// Rcpp has its own stream object which cooperates more nicely with R's i/o
// And as of armadillo and mlpack, we can use this stream object as well.
#if !defined(ARMA_COUT_STREAM)
  #define ARMA_COUT_STREAM Rcpp::Rcout
#endif
#if !defined(ARMA_CERR_STREAM)
  #define ARMA_CERR_STREAM Rcpp::Rcerr
#endif
#if !defined(MLPACK_COUT_STREAM)
  #define MLPACK_COUT_STREAM Rcpp::Rcout
#endif
#if !defined(MLPACK_CERR_STREAM)
  #define MLPACK_CERR_STREAM Rcpp::Rcerr
#endif

// This define makes the R RNG have precedent over the C++11-based
// RNG provided by Armadillo.
#if !defined(ARMA_RNG_ALT)
  #define ARMA_RNG_ALT         RcppArmadillo/rng/Alt_R_RNG.h
#endif

// To suppress warnings related to core/util/arma_util.hpp.
#define MLPACK_CORE_UTIL_ARMA_CONFIG_HPP

// Undefine macro due to macro collision.
#undef Realloc
#undef Free

#endif
