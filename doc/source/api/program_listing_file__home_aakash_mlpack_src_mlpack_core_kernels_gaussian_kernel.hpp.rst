
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_gaussian_kernel.hpp:

Program Listing for File gaussian_kernel.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_gaussian_kernel.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/gaussian_kernel.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_GAUSSIAN_KERNEL_HPP
   #define MLPACK_CORE_KERNELS_GAUSSIAN_KERNEL_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include <mlpack/core/kernels/kernel_traits.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class GaussianKernel
   {
    public:
     GaussianKernel() : bandwidth(1.0), gamma(-0.5)
     { }
   
     GaussianKernel(const double bandwidth) :
         bandwidth(bandwidth),
         gamma(-0.5 * pow(bandwidth, -2.0))
     { }
   
     template<typename VecTypeA, typename VecTypeB>
     double Evaluate(const VecTypeA& a, const VecTypeB& b) const
     {
       // The precalculation of gamma saves us a little computation time.
       return exp(gamma * metric::SquaredEuclideanDistance::Evaluate(a, b));
     }
   
     double Evaluate(const double t) const
     {
       // The precalculation of gamma saves us a little computation time.
       return exp(gamma * std::pow(t, 2.0));
     }
   
     double Gradient(const double t) const {
       return 2 * t * gamma * exp(gamma * std::pow(t, 2.0));
     }
   
     double GradientForSquaredDistance(const double t) const {
       return gamma * exp(gamma * t);
     }
   
     double Normalizer(const size_t dimension)
     {
       return pow(sqrt(2.0 * M_PI) * bandwidth, (double) dimension);
     }
   
     template<typename VecTypeA, typename VecTypeB>
     double ConvolutionIntegral(const VecTypeA& a, const VecTypeB& b)
     {
       return Evaluate(sqrt(metric::SquaredEuclideanDistance::Evaluate(a, b) /
           2.0)) / (Normalizer(a.n_rows) * pow(2.0, (double) a.n_rows / 2.0));
     }
   
   
     double Bandwidth() const { return bandwidth; }
   
     void Bandwidth(const double bandwidth)
     {
       this->bandwidth = bandwidth;
       this->gamma = -0.5 * pow(bandwidth, -2.0);
     }
   
     double Gamma() const { return gamma; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(bandwidth));
       ar(CEREAL_NVP(gamma));
     }
   
    private:
     double bandwidth;
   
     double gamma;
   };
   
   template<>
   class KernelTraits<GaussianKernel>
   {
    public:
     static const bool IsNormalized = true;
     static const bool UsesSquaredDistance = true;
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
