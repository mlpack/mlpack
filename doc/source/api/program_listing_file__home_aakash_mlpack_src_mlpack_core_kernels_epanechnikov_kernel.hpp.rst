
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_epanechnikov_kernel.hpp:

Program Listing for File epanechnikov_kernel.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_epanechnikov_kernel.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/epanechnikov_kernel.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_EPANECHNIKOV_KERNEL_HPP
   #define MLPACK_CORE_KERNELS_EPANECHNIKOV_KERNEL_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/kernels/kernel_traits.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class EpanechnikovKernel
   {
    public:
     EpanechnikovKernel(const double bandwidth = 1.0) :
         bandwidth(bandwidth),
         inverseBandwidthSquared(1.0 / (bandwidth * bandwidth))
     {  }
   
     template<typename VecTypeA, typename VecTypeB>
     double Evaluate(const VecTypeA& a, const VecTypeB& b) const;
   
     double Evaluate(const double distance) const;
   
     double Gradient(const double distance) const;
   
     double GradientForSquaredDistance(const double distanceSquared) const;
     template<typename VecTypeA, typename VecTypeB>
     double ConvolutionIntegral(const VecTypeA& a, const VecTypeB& b);
   
     double Normalizer(const size_t dimension);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     double bandwidth;
     double inverseBandwidthSquared;
   };
   
   template<>
   class KernelTraits<EpanechnikovKernel>
   {
    public:
     static const bool IsNormalized = true;
     static const bool UsesSquaredDistance = true;
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   // Include implementation.
   #include "epanechnikov_kernel_impl.hpp"
   
   #endif
