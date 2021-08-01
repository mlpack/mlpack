
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_mlpack_src_rcpp_mlpack.h:

Program Listing for File rcpp_mlpack.h
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_mlpack_src_rcpp_mlpack.h>` (``/home/aakash/mlpack/src/mlpack/bindings/R/mlpack/src/rcpp_mlpack.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_RCPP_MLPACK_H
   #define MLPACK_BINDINGS_R_RCPP_MLPACK_H
   
   #include <Rcpp.h>
   
   // To suppress Found '__assert_fail', possibly from 'assert' (C).
   #define BOOST_DISABLE_ASSERTS
   
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
     #define ARMA_RNG_ALT         RcppArmadillo/Alt_R_RNG.h
   #endif
   
   // To suppress warnings related to core/util/arma_util.hpp.
   #define MLPACK_CORE_UTIL_ARMA_CONFIG_HPP
   
   // Undefine macro due to macro collision.
   #undef Realloc
   #undef Free
   
   #endif
