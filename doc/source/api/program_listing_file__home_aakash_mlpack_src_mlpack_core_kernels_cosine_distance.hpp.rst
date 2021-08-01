
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_cosine_distance.hpp:

Program Listing for File cosine_distance.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_cosine_distance.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/cosine_distance.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_COSINE_DISTANCE_HPP
   #define MLPACK_CORE_KERNELS_COSINE_DISTANCE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/kernels/kernel_traits.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class CosineDistance
   {
    public:
     template<typename VecTypeA, typename VecTypeB>
     static double Evaluate(const VecTypeA& a, const VecTypeB& b);
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   template<>
   class KernelTraits<CosineDistance>
   {
    public:
     static const bool IsNormalized = true;
   
     static const bool UsesSquaredDistance = false;
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   // Include implementation.
   #include "cosine_distance_impl.hpp"
   
   #endif
