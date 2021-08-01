
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_arma_config_check.hpp:

Program Listing for File arma_config_check.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_arma_config_check.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/arma_config_check.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_ARMA_CONFIG_CHECK_HPP
   #define MLPACK_CORE_UTIL_ARMA_CONFIG_CHECK_HPP
   
   #include "arma_config.hpp"
   
   #ifdef ARMA_64BIT_WORD
     #ifdef MLPACK_ARMA_NO_64BIT_WORD
       #pragma message "mlpack was compiled without ARMA_64BIT_WORD, but you are \
   compiling with ARMA_64BIT_WORD.  This will almost certainly cause irreparable \
   disaster.  Either disable ARMA_64BIT_WORD in your application which is using \
   mlpack, or, recompile mlpack against a version of Armadillo which has \
   ARMA_64BIT_WORD enabled."
     #endif
   #else
     #ifdef MLPACK_ARMA_64BIT_WORD
       #pragma message "mlpack was compiled with ARMA_64BIT_WORD, but you are \
   compiling without ARMA_64BIT_WORD.  This will almost certainly cause \
   irreparable disaster.  Either enable ARMA_64BIT_WORD in your application which \
   is using mlpack, or, recompile mlpack against a version of Armadillo which has \
   ARMA_64BIT_WORD disabled."
     #endif
   #endif
   
   // Check if OpenMP was enabled when mlpack was built.
   #ifdef ARMA_USE_OPENMP
     #ifdef MLPACK_ARMA_DONT_USE_OPENMP
       #pragma message "mlpack was compiled without OpenMP support, but you are \
   compiling with OpenMP support (either -fopenmp or another option).  This will \
   almost certainly cause irreparable disaster.  Either compile your application \
   *without* OpenMP support (i.e. remove -fopenmp or another flag), or, recompile \
   mlpack with OpenMP support."
     #endif
   #else
     #ifdef MLPACK_ARMA_USE_OPENMP
       #pragma message "mlpack was compiled with OpenMP support, but you are \
   compiling without OpenMP support.  This will almost certainly cause \
   irreparable disaster.  Either enable OpenMP support in your application (e.g., \
   add -fopenmp to your compiler command line), or, recompile mlpack *without* \
   OpenMP support."
     #endif
   #endif
   
   #endif
