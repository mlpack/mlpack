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

// Armadillo does not provide an official support for unsigned / signed 8 bits
// integers.
// Since `char` might be represented differently on various hardware.
// We override Armadillo definition for unsigned and signed 8 bits integer to
// use uint8_t / int8_t respectively.
#ifndef ARMA_U8_TYPE
  #define ARMA_U8_TYPE std::uint8_t
#endif

#ifndef ARMA_S8_TYPE
  #define ARMA_S8_TYPE std::int8_t
#endif

// This also includes Rcpp headers along with RcppArmadillo
#include <RcppArmadillo.h>

// Rcpp has its own stream object which cooperates more nicely with R's i/o
// And like armadillo, mlpack can use this stream object as well.
#if !defined(MLPACK_COUT_STREAM)
  #define MLPACK_COUT_STREAM Rcpp::Rcout
#endif
#if !defined(MLPACK_CERR_STREAM)
  #define MLPACK_CERR_STREAM Rcpp::Rcerr
#endif

// The R bindings default to not enabling STB, DR_LIBS or HTTPLIB.
// This can be overriden via package compilerflags, i.e.
//   PKG_CPPFLAGS=-DMLPACK_R_ENABLE_DR_LIBS R CMD INSTALL mlpack_*.tar.gz
// on the command-line, or by editing src/Makevars or ~/.R/Makevars.
#if !defined(MLPACK_R_ENABLE_STB)
  #undef  MLPACK_DISABLE_STB
  #define MLPACK_DISABLE_STB
#endif

#if !defined(MLPACK_R_ENABLE_DR_LIBS)
  #undef  MLPACK_DISABLE_DR_LIBS
  #define MLPACK_DISABLE_DR_LIBS
#endif

#if !defined(MLPACK_R_ENABLE_HTTPLIB)
  #undef  MLPACK_DISABLE_HTTPLIB
  #define MLPACK_DISABLE_HTTPLIB
#endif

#endif
