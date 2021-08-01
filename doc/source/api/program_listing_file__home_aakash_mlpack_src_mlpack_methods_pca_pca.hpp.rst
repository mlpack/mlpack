
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_pca_pca.hpp:

Program Listing for File pca.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_pca_pca.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/pca/pca.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_PCA_PCA_HPP
   #define MLPACK_METHODS_PCA_PCA_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp>
   
   namespace mlpack {
   namespace pca {
   
   template<typename DecompositionPolicy = ExactSVDPolicy>
   class PCA
   {
    public:
     PCA(const bool scaleData = false,
         const DecompositionPolicy& decomposition = DecompositionPolicy());
   
     void Apply(const arma::mat& data,
                arma::mat& transformedData,
                arma::vec& eigVal,
                arma::mat& eigvec);
   
     void Apply(const arma::mat& data,
                arma::mat& transformedData,
                arma::vec& eigVal);
     void Apply(const arma::mat& data,
                arma::mat& transformedData);
   
     double Apply(arma::mat& data, const size_t newDimension);
   
     inline double Apply(arma::mat& data, const int newDimension)
     {
       return Apply(data, size_t(newDimension));
     }
   
     double Apply(arma::mat& data, const double varRetained);
   
     bool ScaleData() const { return scaleData; }
     bool& ScaleData() { return scaleData; }
   
    private:
     void ScaleData(arma::mat& centeredData)
     {
       if (scaleData)
       {
         // Scaling the data is when we reduce the variance of each dimension
         // to 1. We do this by dividing each dimension by its standard
         // deviation.
         arma::vec stdDev = arma::stddev(
             centeredData, 0, 1 /* for each dimension */);
   
         // If there are any zeroes, make them very small.
         for (size_t i = 0; i < stdDev.n_elem; ++i)
           if (stdDev[i] == 0)
             stdDev[i] = 1e-50;
   
         centeredData /= arma::repmat(stdDev, 1, centeredData.n_cols);
       }
     }
   
     bool scaleData;
   
     DecompositionPolicy decomposition;
   }; // class PCA
   
   } // namespace pca
   } // namespace mlpack
   
   // Include implementation.
   #include "pca_impl.hpp"
   
   #endif
