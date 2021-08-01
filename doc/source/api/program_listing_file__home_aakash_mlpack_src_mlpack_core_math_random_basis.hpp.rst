
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_math_random_basis.hpp:

Program Listing for File random_basis.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_math_random_basis.hpp>` (``/home/aakash/mlpack/src/mlpack/core/math/random_basis.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_MATH_RANDOM_BASIS_HPP
   #define MLPACK_CORE_MATH_RANDOM_BASIS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace math {
   
   void RandomBasis(arma::mat& basis, const size_t d);
   
   } // namespace math
   } // namespace mlpack
   
   #endif
