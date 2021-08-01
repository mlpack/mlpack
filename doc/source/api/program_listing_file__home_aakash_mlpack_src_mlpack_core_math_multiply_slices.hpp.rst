
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_math_multiply_slices.hpp:

Program Listing for File multiply_slices.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_math_multiply_slices.hpp>` (``/home/aakash/mlpack/src/mlpack/core/math/multiply_slices.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_MATH_MULTIPLY_SLICES_HPP
   #define MLPACK_CORE_MATH_MULTIPLY_SLICES_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace math  {
   
   template <typename CubeType>
   CubeType MultiplyCube2Cube(const CubeType& cubeA,
                              const CubeType& cubeB,
                              const bool aTranspose = false,
                              const bool bTranspose = false);
   template <typename MatType, typename CubeType>
   CubeType MultiplyMat2Cube(const MatType& matA,
                             const CubeType& cubeB,
                             const bool aTranspose = false,
                             const bool bTranspose = false);
   template <typename CubeType, typename MatType>
   CubeType MultiplyCube2Mat(const CubeType& cubeA,
                             const MatType& matB,
                             const bool aTranspose = false,
                             const bool bTranspose = false);
   
   } // namespace math
   } // namespace mlpack
   
   // Include implementation.
   #include "multiply_slices_impl.hpp"
   
   #endif
