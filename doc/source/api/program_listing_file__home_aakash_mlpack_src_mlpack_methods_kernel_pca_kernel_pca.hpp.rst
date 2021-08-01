
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kernel_pca_kernel_pca.hpp:

Program Listing for File kernel_pca.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kernel_pca_kernel_pca.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kernel_pca/kernel_pca.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_HPP
   #define MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/kernel_pca/kernel_rules/naive_method.hpp>
   
   namespace mlpack {
   namespace kpca {
   
   template <
     typename KernelType,
     typename KernelRule = NaiveKernelRule<KernelType>
   >
   class KernelPCA
   {
    public:
     KernelPCA(const KernelType kernel = KernelType(),
               const bool centerTransformedData = false);
   
     void Apply(const arma::mat& data,
                arma::mat& transformedData,
                arma::vec& eigval,
                arma::mat& eigvec,
                const size_t newDimension);
   
     void Apply(const arma::mat& data,
                arma::mat& transformedData,
                arma::vec& eigval,
                arma::mat& eigvec);
   
     void Apply(const arma::mat& data,
                arma::mat& transformedData,
                arma::vec& eigval);
   
     void Apply(arma::mat& data, const size_t newDimension);
   
     const KernelType& Kernel() const { return kernel; }
     KernelType& Kernel() { return kernel; }
   
     bool CenterTransformedData() const { return centerTransformedData; }
     bool& CenterTransformedData() { return centerTransformedData; }
   
    private:
     KernelType kernel;
     bool centerTransformedData;
   }; // class KernelPCA
   
   } // namespace kpca
   } // namespace mlpack
   
   // Include implementation.
   #include "kernel_pca_impl.hpp"
   
   #endif // MLPACK_METHODS_KERNEL_PCA_KERNEL_PCA_HPP
