
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_math_log_add.hpp:

Program Listing for File log_add.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_math_log_add.hpp>` (``/home/aakash/mlpack/src/mlpack/core/math/log_add.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_MATH_LOG_ADD_HPP
   #define MLPACK_CORE_MATH_LOG_ADD_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace math {
   
   template<typename T>
   T LogAdd(T x, T y);
   
   template<typename T>
   typename T::elem_type AccuLog(const T& x);
   
   template<typename T, bool InPlace = false>
   void LogSumExp(const T& x, arma::Col<typename T::elem_type>& y);
   
   template<typename T, bool InPlace = false>
   void LogSumExpT(const T& x, arma::Col<typename T::elem_type>& y);
   
   } // namespace math
   } // namespace mlpack
   
   // Include implementation.
   #include "log_add_impl.hpp"
   
   #endif
