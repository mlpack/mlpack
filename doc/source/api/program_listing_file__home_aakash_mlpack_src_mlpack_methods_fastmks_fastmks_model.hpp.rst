
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_fastmks_fastmks_model.hpp:

Program Listing for File fastmks_model.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_fastmks_fastmks_model.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/fastmks/fastmks_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_FASTMKS_FASTMKS_MODEL_HPP
   #define MLPACK_METHODS_FASTMKS_FASTMKS_MODEL_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "fastmks.hpp"
   #include <mlpack/core/kernels/kernel_traits.hpp>
   #include <mlpack/core/kernels/linear_kernel.hpp>
   #include <mlpack/core/kernels/polynomial_kernel.hpp>
   #include <mlpack/core/kernels/cosine_distance.hpp>
   #include <mlpack/core/kernels/gaussian_kernel.hpp>
   #include <mlpack/core/kernels/epanechnikov_kernel.hpp>
   #include <mlpack/core/kernels/hyperbolic_tangent_kernel.hpp>
   #include <mlpack/core/kernels/laplacian_kernel.hpp>
   #include <mlpack/core/kernels/pspectrum_string_kernel.hpp>
   #include <mlpack/core/kernels/spherical_kernel.hpp>
   #include <mlpack/core/kernels/triangular_kernel.hpp>
   
   namespace mlpack {
   namespace fastmks {
   
   class FastMKSModel
   {
    public:
     enum KernelTypes
     {
       LINEAR_KERNEL,
       POLYNOMIAL_KERNEL,
       COSINE_DISTANCE,
       GAUSSIAN_KERNEL,
       EPANECHNIKOV_KERNEL,
       TRIANGULAR_KERNEL,
       HYPTAN_KERNEL
     };
   
     FastMKSModel(const int kernelType = LINEAR_KERNEL);
   
     FastMKSModel(const FastMKSModel& other);
   
     FastMKSModel(FastMKSModel&& other);
   
     FastMKSModel& operator=(const FastMKSModel& other);
   
     FastMKSModel& operator=(FastMKSModel&& other);
   
     ~FastMKSModel();
   
     template<typename TKernelType>
     void BuildModel(util::Timers& timers,
                     arma::mat&& referenceData,
                     TKernelType& kernel,
                     const bool singleMode,
                     const bool naive,
                     const double base);
   
     bool Naive() const;
     bool& Naive();
   
     bool SingleMode() const;
     bool& SingleMode();
   
     int KernelType() const { return kernelType; }
     int& KernelType() { return kernelType; }
   
     void Search(util::Timers& timers,
                 const arma::mat& querySet,
                 const size_t k,
                 arma::Mat<size_t>& indices,
                 arma::mat& kernels,
                 const double base);
   
     void Search(util::Timers& timers,
                 const size_t k,
                 arma::Mat<size_t>& indices,
                 arma::mat& kernels);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     int kernelType;
   
     FastMKS<kernel::LinearKernel>* linear;
     FastMKS<kernel::PolynomialKernel>* polynomial;
     FastMKS<kernel::CosineDistance>* cosine;
     FastMKS<kernel::GaussianKernel>* gaussian;
     FastMKS<kernel::EpanechnikovKernel>* epan;
     FastMKS<kernel::TriangularKernel>* triangular;
     FastMKS<kernel::HyperbolicTangentKernel>* hyptan;
   
     template<typename FastMKSType>
     void Search(util::Timers& timers,
                 FastMKSType& f,
                 const arma::mat& querySet,
                 const size_t k,
                 arma::Mat<size_t>& indices,
                 arma::mat& kernels,
                 const double base);
   };
   
   } // namespace fastmks
   } // namespace mlpack
   
   #include "fastmks_model_impl.hpp"
   
   #endif
