
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_scaler_methods_max_abs_scaler.hpp:

Program Listing for File max_abs_scaler.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_scaler_methods_max_abs_scaler.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/scaler_methods/max_abs_scaler.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_MAX_ABS_SCALE_HPP
   #define MLPACK_CORE_DATA_MAX_ABS_SCALE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   
   class MaxAbsScaler
   {
    public:
     template<typename MatType>
     void Fit(const MatType& input)
     {
       itemMin = arma::min(input, 1);
       itemMax = arma::max(input, 1);
       scale = arma::max(arma::abs(itemMin), arma::abs(itemMax));
       // Handling zeros in scale vector.
       scale.for_each([](arma::vec::elem_type& val) { val =
           (val == 0) ? 1 : val; });
     }
   
     template<typename MatType>
     void Transform(const MatType& input, MatType& output)
     {
       if (scale.is_empty())
       {
         throw std::runtime_error("Call Fit() before Transform(), please"
           " refer to the documentation.");
       }
       output.copy_size(input);
       output = input.each_col() / scale;
     }
   
     template<typename MatType>
     void InverseTransform(const MatType& input, MatType& output)
     {
       output.copy_size(input);
       output = input.each_col() % scale;
     }
   
     const arma::vec& ItemMin() const { return itemMin; }
     const arma::vec& ItemMax() const { return itemMax; }
     const arma::vec& Scale() const { return scale; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(itemMin));
       ar(CEREAL_NVP(itemMax));
       ar(CEREAL_NVP(scale));
     }
    private:
     // Vector which holds minimum of each feature.
     arma::vec itemMin;
     // Vector which holds maximum of each feature.
     arma::vec itemMax;
     // Vector which is used to scale up each feature.
     arma::vec scale;
   }; // class MaxAbsScaler
   
   } // namespace data
   } // namespace mlpack
   
   #endif
