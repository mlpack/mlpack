
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_scaler_methods_zca_whitening.hpp:

Program Listing for File zca_whitening.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_scaler_methods_zca_whitening.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/scaler_methods/zca_whitening.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_ZCA_WHITENING_SCALE_HPP
   #define MLPACK_CORE_DATA_ZCA_WHITENING_SCALE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/lin_alg.hpp>
   #include <mlpack/core/data/scaler_methods/pca_whitening.hpp>
   
   namespace mlpack {
   namespace data {
   
   class ZCAWhitening
   {
    public:
     ZCAWhitening(double eps = 0.00005) : pca(eps) { }
   
     template<typename MatType>
     void Fit(const MatType& input)
     {
       pca.Fit(input);
     }
   
     template<typename MatType>
     void Transform(const MatType& input, MatType& output)
     {
       pca.Transform(input, output);
       output = pca.EigenVectors() * output;
     }
   
     template<typename MatType>
     void InverseTransform(const MatType& input, MatType& output)
     {
       output = inv(pca.EigenVectors()) * arma::diagmat(arma::sqrt(
           pca.EigenValues())) * inv(pca.EigenVectors().t()) * input;
       output = (output.each_col() + pca.ItemMean());
     }
   
     const arma::vec& ItemMean() const { return pca.ItemMean(); }
     const arma::vec& EigenValues() const { return pca.EigenValues(); }
     const arma::mat& EigenVectors() const { return pca.EigenVectors(); }
     double Epsilon() const { return pca.Epsilon(); }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(pca));
     }
   
    private:
     // A pointer to PcaWhitening Class.
     PCAWhitening pca;
   }; // class ZCAWhitening
   
   } // namespace data
   } // namespace mlpack
   
   #endif
