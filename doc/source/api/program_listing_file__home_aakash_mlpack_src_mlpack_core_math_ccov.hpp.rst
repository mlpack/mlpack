
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_math_ccov.hpp:

Program Listing for File ccov.hpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_math_ccov.hpp>` (``/home/aakash/mlpack/src/mlpack/core/math/ccov.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_MATH_CCOV_HPP
   #define MLPACK_CORE_MATH_CCOV_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace math  {
   
   template<typename eT>
   inline
   arma::Mat<eT>
   ColumnCovariance(const arma::Mat<eT>& A, const size_t norm_type = 0);
   
   template<typename T>
   inline
   arma::Mat< std::complex<T> >
   ColumnCovariance(const arma::Mat< std::complex<T> >& A,
        const size_t norm_type = 0);
   
   } // namespace math
   } // namespace mlpack
   
   // Include implementation
   #include "ccov_impl.hpp"
   
   #endif // MLPACK_CORE_MATH_CCOV_HPP
