
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_cauchy_kernel.hpp:

Program Listing for File cauchy_kernel.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_cauchy_kernel.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/cauchy_kernel.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_CAUCHY_KERNEL_HPP
   #define MLPACK_CORE_KERNELS_CAUCHY_KERNEL_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include <mlpack/core/kernels/kernel_traits.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class CauchyKernel
   {
    public:
     CauchyKernel(double bandwidth = 1.0) : bandwidth(bandwidth)
     { }
   
     template<typename VecTypeA, typename VecTypeB>
     double Evaluate(const VecTypeA& a, const VecTypeB& b)
     {
       return 1 / (1 + (
           std::pow(metric::EuclideanDistance::Evaluate(a, b) / bandwidth, 2)));
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(bandwidth));
     }
   
    private:
     double bandwidth;
   };
   
   template<>
   class KernelTraits<CauchyKernel>
   {
    public:
     static const bool IsNormalized = true;
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
