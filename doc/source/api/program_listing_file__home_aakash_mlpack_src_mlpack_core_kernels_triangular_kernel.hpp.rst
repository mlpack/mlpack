
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_triangular_kernel.hpp:

Program Listing for File triangular_kernel.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_triangular_kernel.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/triangular_kernel.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_TRIANGULAR_KERNEL_HPP
   #define MLPACK_CORE_KERNELS_TRIANGULAR_KERNEL_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class TriangularKernel
   {
    public:
     TriangularKernel(const double bandwidth = 1.0) : bandwidth(bandwidth) { }
   
     template<typename VecTypeA, typename VecTypeB>
     double Evaluate(const VecTypeA& a, const VecTypeB& b) const
     {
       return std::max(0.0, (1 - metric::EuclideanDistance::Evaluate(a, b) /
           bandwidth));
     }
   
     double Evaluate(const double distance) const
     {
       return std::max(0.0, (1 - distance) / bandwidth);
     }
   
     double Gradient(const double distance) const
     {
       if (distance < 1)
       {
         return -1.0 / bandwidth;
       }
       else if (distance > 1)
       {
         return 0;
       }
       else
       {
         return arma::datum::nan;
       }
     }
   
     double Bandwidth() const { return bandwidth; }
     double& Bandwidth() { return bandwidth; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(bandwidth));
     }
   
    private:
     double bandwidth;
   };
   
   template<>
   class KernelTraits<TriangularKernel>
   {
    public:
     static const bool IsNormalized = true;
     static const bool UsesSquaredDistance = false;
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
