
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_scaler_methods_min_max_scaler.hpp:

Program Listing for File min_max_scaler.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_scaler_methods_min_max_scaler.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/scaler_methods/min_max_scaler.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_SCALE_HPP
   #define MLPACK_CORE_DATA_SCALE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   
   class MinMaxScaler
   {
    public:
     MinMaxScaler(const double min = 0, const double max = 1)
     {
       scaleMin = min;
       scaleMax = max;
       // Ensure scaleMin is smaller than scaleMax.
       if (scaleMin > scaleMax)
       {
         throw std::runtime_error("Range is not appropriate");
       }
     }
   
     template<typename MatType>
     void Fit(const MatType& input)
     {
       itemMin = arma::min(input, 1);
       itemMax = arma::max(input, 1);
       scale = itemMax - itemMin;
       // Handle zeros in scale vector.
       scale.for_each([](arma::vec::elem_type& val) { val =
           (val == 0) ? 1 : val; });
       scale = (scaleMax - scaleMin) / scale;
       scalerowmin.copy_size(itemMin);
       scalerowmin.fill(scaleMin);
       scalerowmin = scalerowmin - itemMin % scale;
     }
   
     template<typename MatType>
     void Transform(const MatType& input, MatType& output)
     {
       if (scalerowmin.is_empty() || scale.is_empty())
       {
         throw std::runtime_error("Call Fit() before Transform(), please"
             " refer to the documentation.");
       }
       output.copy_size(input);
       output = (input.each_col() % scale).each_col() + scalerowmin;
     }
   
     template<typename MatType>
     void InverseTransform(const MatType& input, MatType& output)
     {
       output.copy_size(input);
       output = (input.each_col() - scalerowmin).each_col() / scale;
     }
   
     const arma::vec& ItemMin() const { return itemMin; }
     const arma::vec& ItemMax() const { return itemMax; }
     const arma::vec& Scale() const { return scale; }
     double ScaleMax() const { return scaleMax; }
     double ScaleMin() const { return scaleMin; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(itemMin));
       ar(CEREAL_NVP(itemMax));
       ar(CEREAL_NVP(scale));
       ar(CEREAL_NVP(scaleMin));
       ar(CEREAL_NVP(scaleMax));
       ar(CEREAL_NVP(scalerowmin));
     }
   
    private:
     // Vector which holds minimum of each feature.
     arma::vec itemMin;
     // Vector which holds maximum of each feature.
     arma::vec itemMax;
     // Scale vector which is used to scale up each feature.
     arma::vec scale;
     // Lower value for range.
     double scaleMin;
     // Upper value for range.
     double scaleMax;
     // Column vector of scalemin
     arma::vec scalerowmin;
   }; // class MinMaxScaler
   
   } // namespace data
   } // namespace mlpack
   
   #endif
