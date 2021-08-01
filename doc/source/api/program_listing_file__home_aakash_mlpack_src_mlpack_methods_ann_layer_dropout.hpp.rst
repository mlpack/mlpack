
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_dropout.hpp:

Program Listing for File dropout.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_dropout.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/dropout.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_DROPOUT_HPP
   #define MLPACK_METHODS_ANN_LAYER_DROPOUT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   
   template<typename InputDataType = arma::mat,
            typename OutputDataType = arma::mat>
   class Dropout
   {
    public:
     Dropout(const double ratio = 0.5);
   
     Dropout(const Dropout& layer);
   
     Dropout(const Dropout&&);
   
     Dropout& operator=(const Dropout& layer);
   
     Dropout& operator=(Dropout&& layer);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     bool Deterministic() const { return deterministic; }
     bool& Deterministic() { return deterministic; }
   
     double Ratio() const { return ratio; }
   
     void Ratio(const double r)
     {
       ratio = r;
       scale = 1.0 / (1.0 - ratio);
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     OutputDataType mask;
   
     double ratio;
   
     double scale;
   
     bool deterministic;
   }; // class Dropout
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "dropout_impl.hpp"
   
   #endif
