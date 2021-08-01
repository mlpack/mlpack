
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_scaler_methods_mean_normalization.hpp:

Program Listing for File mean_normalization.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_scaler_methods_mean_normalization.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/scaler_methods/mean_normalization.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_MEAN_NORMALIZATION_HPP
   #define MLPACK_CORE_DATA_MEAN_NORMALIZATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   
   class MeanNormalization
   {
    public:
     template<typename MatType>
     void Fit(const MatType& input)
     {
       itemMean = arma::mean(input, 1);
       itemMin = arma::min(input, 1);
       itemMax = arma::max(input, 1);
       scale = itemMax - itemMin;
       // Handling zeros in scale vector.
       scale.for_each([](arma::vec::elem_type& val) { val =
           (val == 0) ? 1 : val; });
     }
   
     template<typename MatType>
     void Transform(const MatType& input, MatType& output)
     {
       if (itemMean.is_empty() || scale.is_empty())
       {
         throw std::runtime_error("Call Fit() before Transform(), please"
           " refer to the documentation.");
       }
       output.copy_size(input);
       output = (input.each_col() - itemMean).each_col() / scale;
     }
   
     template<typename MatType>
     void InverseTransform(const MatType& input, MatType& output)
     {
       output.copy_size(input);
       output = (input.each_col() % scale).each_col() + itemMean;
     }
   
     const arma::vec& ItemMean() const { return itemMean; }
     const arma::vec& ItemMin() const { return itemMin; }
     const arma::vec& ItemMax() const { return itemMax; }
     const arma::vec& Scale() const { return scale; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(itemMin));
       ar(CEREAL_NVP(itemMax));
       ar(CEREAL_NVP(scale));
       ar(CEREAL_NVP(itemMean));
     }
   
    private:
     // Vector which holds mean of each feature.
     arma::vec itemMean;
     // Vector which holds minimum of each feature.
     arma::vec itemMin;
     // Vector which holds maximum of each feature.
     arma::vec itemMax;
     // Vector which is used to scale up each feature.
     arma::vec scale;
   }; // class MeanNormalization
   
   } // namespace data
   } // namespace mlpack
   
   #endif
