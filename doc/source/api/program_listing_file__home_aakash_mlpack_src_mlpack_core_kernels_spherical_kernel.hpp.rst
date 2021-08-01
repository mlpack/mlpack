
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_spherical_kernel.hpp:

Program Listing for File spherical_kernel.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_spherical_kernel.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/spherical_kernel.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_SPHERICAL_KERNEL_HPP
   #define MLPACK_CORE_KERNELS_SPHERICAL_KERNEL_HPP
   
   #include <boost/math/special_functions/gamma.hpp>
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class SphericalKernel
   {
    public:
     SphericalKernel(const double bandwidth = 1.0) :
       bandwidth(bandwidth),
       bandwidthSquared(std::pow(bandwidth, 2.0))
     { /* Nothing to do. */ }
   
     template<typename VecTypeA, typename VecTypeB>
     double Evaluate(const VecTypeA& a, const VecTypeB& b) const
     {
       return
           (metric::SquaredEuclideanDistance::Evaluate(a, b) <= bandwidthSquared) ?
           1.0 : 0.0;
     }
     template<typename VecTypeA, typename VecTypeB>
     double ConvolutionIntegral(const VecTypeA& a, const VecTypeB& b) const
     {
       double distance = sqrt(metric::SquaredEuclideanDistance::Evaluate(a, b));
       if (distance >= 2.0 * bandwidth)
       {
         return 0.0;
       }
       double volumeSquared = pow(Normalizer(a.n_rows), 2.0);
   
       switch (a.n_rows)
       {
         case 1:
           return 1.0 / volumeSquared * (2.0 * bandwidth - distance);
         case 2:
           return 1.0 / volumeSquared *
             (2.0 * bandwidth * bandwidth * acos(distance/(2.0 * bandwidth)) -
             distance / 4.0 * sqrt(4.0*bandwidth*bandwidth-distance*distance));
         default:
           Log::Fatal << "The spherical kernel does not support convolution\
             integrals above dimension two, yet..." << std::endl;
           return -1.0;
       }
     }
     double Normalizer(size_t dimension) const
     {
       return pow(bandwidth, (double) dimension) * pow(M_PI, dimension / 2.0) /
           std::tgamma(dimension / 2.0 + 1.0);
     }
   
     double Evaluate(const double t) const
     {
       return (t <= bandwidth) ? 1.0 : 0.0;
     }
     double Gradient(double t)
     {
       return t == bandwidth ? arma::datum::nan : 0.0;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(bandwidth));
       ar(CEREAL_NVP(bandwidthSquared));
     }
   
    private:
     double bandwidth;
     double bandwidthSquared;
   };
   
   template<>
   class KernelTraits<SphericalKernel>
   {
    public:
     static const bool IsNormalized = true;
     static const bool UsesSquaredDistance = false;
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
