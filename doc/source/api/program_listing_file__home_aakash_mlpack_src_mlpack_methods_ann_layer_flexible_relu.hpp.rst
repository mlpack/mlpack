
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_flexible_relu.hpp:

Program Listing for File flexible_relu.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_flexible_relu.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/flexible_relu.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_FLEXIBLERELU_HPP
   #define MLPACK_METHODS_ANN_LAYER_FLEXIBLERELU_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class FlexibleReLU
   {
    public:
     FlexibleReLU(const double alpha = 0);
   
     void Reset();
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
     template<typename DataType>
     void Backward(const DataType& input, const DataType& gy, DataType& g);
   
     template<typename eT>
     void Gradient(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& error,
                   arma::Mat<eT>& gradient);
   
     OutputDataType const& Parameters() const { return alpha; }
     OutputDataType& Parameters() { return alpha; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta;}
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
     double const& Alpha() const { return alpha; }
     double& Alpha() { return alpha; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version*/);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     OutputDataType alpha;
   
     OutputDataType gradient;
   
     double userAlpha;
   }; // class FlexibleReLU
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation
   #include "flexible_relu_impl.hpp"
   
   #endif
