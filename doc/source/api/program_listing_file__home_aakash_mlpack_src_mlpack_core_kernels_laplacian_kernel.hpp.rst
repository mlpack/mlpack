
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_laplacian_kernel.hpp:

Program Listing for File laplacian_kernel.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_laplacian_kernel.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/laplacian_kernel.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_LAPLACIAN_KERNEL_HPP
   #define MLPACK_CORE_KERNELS_LAPLACIAN_KERNEL_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class LaplacianKernel
   {
    public:
     LaplacianKernel() : bandwidth(1.0)
     { }
   
     LaplacianKernel(double bandwidth) :
         bandwidth(bandwidth)
     { }
   
     template<typename VecTypeA, typename VecTypeB>
     double Evaluate(const VecTypeA& a, const VecTypeB& b) const
     {
       // The precalculation of gamma saves us a little computation time.
       return exp(-metric::EuclideanDistance::Evaluate(a, b) / bandwidth);
     }
   
     double Evaluate(const double t) const
     {
       // The precalculation of gamma saves us a little computation time.
       return exp(-t / bandwidth);
     }
   
     double Gradient(const double t) const  {
       return exp(-t / bandwidth) / -bandwidth;
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
   class KernelTraits<LaplacianKernel>
   {
    public:
     static const bool IsNormalized = true;
     static const bool UsesSquaredDistance = false;
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
