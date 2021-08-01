
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_pca_pca_impl.hpp:

Program Listing for File pca_impl.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_pca_pca_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/pca/pca_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_PCA_PCA_IMPL_HPP
   #define MLPACK_METHODS_PCA_PCA_IMPL_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/lin_alg.hpp>
   #include "pca.hpp"
   
   namespace mlpack {
   namespace pca {
   
   template<typename DecompositionPolicy>
   PCA<DecompositionPolicy>::PCA(
       const bool scaleData, const DecompositionPolicy& decomposition) :
       scaleData(scaleData),
       decomposition(decomposition)
   { }
   
   template<typename DecompositionPolicy>
   void PCA<DecompositionPolicy>::Apply(const arma::mat& data,
                                        arma::mat& transformedData,
                                        arma::vec& eigVal,
                                        arma::mat& eigvec)
   {
     Timer::Start("pca");
   
     // Center the data into a temporary matrix.
     arma::mat centeredData;
     math::Center(data, centeredData);
   
     // Scale the data if the user ask for.
     ScaleData(centeredData);
   
     decomposition.Apply(data, centeredData, transformedData, eigVal, eigvec,
         data.n_rows);
   
     Timer::Stop("pca");
   }
   
   template<typename DecompositionPolicy>
   void PCA<DecompositionPolicy>::Apply(const arma::mat& data,
                                        arma::mat& transformedData,
                                        arma::vec& eigVal)
   {
     arma::mat eigvec;
     Apply(data, transformedData, eigVal, eigvec);
   }
   
   template<typename DecompositionPolicy>
   void PCA<DecompositionPolicy>::Apply(const arma::mat& data,
                                        arma::mat& transformedData)
   {
     arma::mat eigvec;
     arma::vec eigVal;
     Apply(data, transformedData, eigVal, eigvec);
   }
   
   template<typename DecompositionPolicy>
   double PCA<DecompositionPolicy>::Apply(arma::mat& data,
                                          const size_t newDimension)
   {
     // Parameter validation.
     if (newDimension == 0)
       Log::Fatal << "PCA::Apply(): newDimension (" << newDimension << ") cannot "
           << "be zero!" << std::endl;
     if (newDimension > data.n_rows)
       Log::Fatal << "PCA::Apply(): newDimension (" << newDimension << ") cannot "
           << "be greater than the existing dimensionality of the data ("
           << data.n_rows << ")!" << std::endl;
   
     arma::mat eigvec;
     arma::vec eigVal;
   
     Timer::Start("pca");
   
     // Center the data into a temporary matrix.
     arma::mat centeredData;
     math::Center(data, centeredData);
   
     // Scale the data if the user ask for.
     ScaleData(centeredData);
   
     decomposition.Apply(data, centeredData, data, eigVal, eigvec, newDimension);
   
     if (newDimension < eigvec.n_rows)
       // Drop unnecessary rows.
       data.shed_rows(newDimension, data.n_rows - 1);
   
     // The svd method returns only non-zero eigenvalues so we have to calculate
     // the right dimension before calculating the amount of variance retained.
     double eigDim = std::min(newDimension - 1, (size_t) eigVal.n_elem - 1);
   
     Timer::Stop("pca");
   
     // Calculate the total amount of variance retained.
     return (sum(eigVal.subvec(0, eigDim)) / sum(eigVal));
   }
   
   template<typename DecompositionPolicy>
   double PCA<DecompositionPolicy>::Apply(arma::mat& data,
                                          const double varRetained)
   {
     // Parameter validation.
     if (varRetained < 0)
       Log::Fatal << "PCA::Apply(): varRetained (" << varRetained << ") must be "
           << "greater than or equal to 0." << std::endl;
     if (varRetained > 1)
       Log::Fatal << "PCA::Apply(): varRetained (" << varRetained << ") should be "
           << "less than or equal to 1." << std::endl;
   
     arma::mat eigvec;
     arma::vec eigVal;
   
     Apply(data, data, eigVal, eigvec);
   
     // Calculate the dimension we should keep.
     size_t newDimension = 0;
     double varSum = 0.0;
     eigVal /= arma::sum(eigVal); // Normalize eigenvalues.
     while ((varSum < varRetained) && (newDimension < eigVal.n_elem))
     {
       varSum += eigVal[newDimension];
       ++newDimension;
     }
   
     // varSum is the actual variance we will retain.
     if (newDimension < eigVal.n_elem)
       data.shed_rows(newDimension, data.n_rows - 1);
   
     return varSum;
   }
   
   } // namespace pca
   } // namespace mlpack
   
   #endif
