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

#include <RcppArmadillo.h>

// Rcpp has its own stream object which cooperates more nicely with R's i/o
// And like armadillo, mlpack can use this stream object as well.
#if !defined(MLPACK_COUT_STREAM)
  #define MLPACK_COUT_STREAM Rcpp::Rcout
#endif
#if !defined(MLPACK_CERR_STREAM)
  #define MLPACK_CERR_STREAM Rcpp::Rcerr
#endif

#endif
