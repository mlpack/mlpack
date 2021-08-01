
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_scaler_methods_standard_scaler.hpp:

Program Listing for File standard_scaler.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_scaler_methods_standard_scaler.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/scaler_methods/standard_scaler.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_STANDARD_SCALE_HPP
   #define MLPACK_CORE_DATA_STANDARD_SCALE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   
   class StandardScaler
   {
    public:
     template<typename MatType>
     void Fit(const MatType& input)
     {
       itemMean = arma::mean(input, 1);
       itemStdDev = arma::stddev(input, 1, 1);
       // Handle zeros in scale vector.
       itemStdDev.for_each([](arma::vec::elem_type& val) { val =
           (val == 0) ? 1 : val; });
     }
   
     template<typename MatType>
     void Transform(const MatType& input, MatType& output)
     {
       if (itemMean.is_empty() || itemStdDev.is_empty())
       {
         throw std::runtime_error("Call Fit() before Transform(), please"
           " refer to the documentation.");
       }
       output.copy_size(input);
       output = (input.each_col() - itemMean).each_col() / itemStdDev;
     }
   
     template<typename MatType>
     void InverseTransform(const MatType& input, MatType& output)
     {
       output.copy_size(input);
       output = (input.each_col() % itemStdDev).each_col() + itemMean;
     }
   
     const arma::vec& ItemMean() const { return itemMean; }
     const arma::vec& ItemStdDev() const { return itemStdDev; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(itemMean));
       ar(CEREAL_NVP(itemStdDev));
     }
   
    private:
     // Vector which holds mean of each feature.
     arma::vec itemMean;
     // Vector which holds standard devation of each feature.
     arma::vec itemStdDev;
   }; // class StandardScaler
   
   } // namespace data
   } // namespace mlpack
   
   #endif
