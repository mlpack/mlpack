
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_scaler_methods_pca_whitening.hpp:

Program Listing for File pca_whitening.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_scaler_methods_pca_whitening.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/scaler_methods/pca_whitening.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_PCA_WHITENING_SCALE_HPP
   #define MLPACK_CORE_DATA_PCA_WHITENING_SCALE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/lin_alg.hpp>
   #include <mlpack/core/math/ccov.hpp>
   
   namespace mlpack {
   namespace data {
   
   class PCAWhitening
   {
    public:
     PCAWhitening(double eps = 0.00005)
     {
       epsilon = eps;
       // Ensure scaleMin is smaller than scaleMax.
       if (epsilon < 0)
       {
         throw std::runtime_error("Regularization parameter is not correct");
       }
     }
   
     template<typename MatType>
     void Fit(const MatType& input)
     {
       itemMean = arma::mean(input, 1);
       // Get eigenvectors and eigenvalues of covariance of input matrix.
       eig_sym(eigenValues, eigenVectors, mlpack::math::ColumnCovariance(
           input.each_col() - itemMean));
       eigenValues += epsilon;
     }
   
     template<typename MatType>
     void Transform(const MatType& input, MatType& output)
     {
       if (eigenValues.is_empty() || eigenVectors.is_empty())
       {
         throw std::runtime_error("Call Fit() before Transform(), please"
             " refer to the documentation.");
       }
       output.copy_size(input);
       output = (input.each_col() - itemMean);
       output = arma::diagmat(1.0 / (arma::sqrt(eigenValues))) * eigenVectors.t()
           * output;
     }
   
     template<typename MatType>
     void InverseTransform(const MatType& input, MatType& output)
     {
       output = arma::diagmat(arma::sqrt(eigenValues)) * inv(eigenVectors.t())
           * input;
       output = (output.each_col() + itemMean);
     }
   
     const arma::vec& ItemMean() const { return itemMean; }
     const arma::vec& EigenValues() const { return eigenValues; }
     const arma::mat& EigenVectors() const { return eigenVectors; }
     const double& Epsilon() const { return epsilon; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(eigenValues));
       ar(CEREAL_NVP(eigenVectors));
       ar(CEREAL_NVP(itemMean));
       ar(CEREAL_NVP(epsilon));
     }
   
    private:
     // Vector which holds mean of each feature.
     arma::vec itemMean;
     // Mat which hold the eigenvectors.
     arma::mat eigenVectors;
     // Regularization Paramter.
     double epsilon;
     // Vector which hold the eigenvalues.
     arma::vec eigenValues;
   }; // class PCAWhitening
   
   } // namespace data
   } // namespace mlpack
   
   #endif
