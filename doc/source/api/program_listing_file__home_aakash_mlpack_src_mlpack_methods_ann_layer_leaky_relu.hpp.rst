
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_leaky_relu.hpp:

Program Listing for File leaky_relu.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_leaky_relu.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/leaky_relu.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_LEAKYRELU_HPP
   #define MLPACK_METHODS_ANN_LAYER_LEAKYRELU_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class LeakyReLU
   {
    public:
     LeakyReLU(const double alpha = 0.03);
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
     template<typename DataType>
     void Backward(const DataType& input, const DataType& gy, DataType& g);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     double const& Alpha() const { return alpha; }
     double& Alpha() { return alpha; }
   
     size_t WeightSize() const { return 0; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     double alpha;
   }; // class LeakyReLU
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "leaky_relu_impl.hpp"
   
   #endif
