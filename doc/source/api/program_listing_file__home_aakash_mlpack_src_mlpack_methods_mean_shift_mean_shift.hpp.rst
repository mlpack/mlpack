
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_mean_shift_mean_shift.hpp:

Program Listing for File mean_shift.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_mean_shift_mean_shift.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/mean_shift/mean_shift.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP
   #define MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/kernels/gaussian_kernel.hpp>
   #include <mlpack/core/kernels/kernel_traits.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   #include <boost/utility.hpp>
   
   namespace mlpack {
   namespace meanshift  {
   
   template<bool UseKernel = false,
            typename KernelType = kernel::GaussianKernel,
            typename MatType = arma::mat>
   class MeanShift
   {
    public:
     MeanShift(const double radius = 0,
               const size_t maxIterations = 1000,
               const KernelType kernel = KernelType());
   
     double EstimateRadius(const MatType& data, const double ratio = 0.2);
   
     void Cluster(const MatType& data,
                  arma::Row<size_t>& assignments,
                  arma::mat& centroids,
                  bool forceConvergence = true,
                  bool useSeeds = true);
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     double Radius() const { return radius; }
     void Radius(double radius);
   
     const KernelType& Kernel() const { return kernel; }
     KernelType& Kernel() { return kernel; }
   
    private:
     void GenSeeds(const MatType& data,
                   const double binSize,
                   const int minFreq,
                   MatType& seeds);
   
     template<bool ApplyKernel = UseKernel>
     typename std::enable_if<ApplyKernel, bool>::type
     CalculateCentroid(const MatType& data,
                       const std::vector<size_t>& neighbors,
                       const std::vector<double>& distances,
                       arma::colvec& centroid);
   
     template<bool ApplyKernel = UseKernel>
     typename std::enable_if<!ApplyKernel, bool>::type
     CalculateCentroid(const MatType& data,
                       const std::vector<size_t>& neighbors,
                       const std::vector<double>&, /*unused*/
                       arma::colvec& centroid);
   
     double radius;
   
     size_t maxIterations;
   
     KernelType kernel;
   };
   
   } // namespace meanshift
   } // namespace mlpack
   
   // Include implementation.
   #include "mean_shift_impl.hpp"
   
   #endif // MLPACK_METHODS_MEAN_SHIFT_MEAN_SHIFT_HPP
