
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_leaky_relu_impl.hpp:

Program Listing for File leaky_relu_impl.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_leaky_relu_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/leaky_relu_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_LEAKYRELU_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_LEAKYRELU_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "leaky_relu.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   LeakyReLU<InputDataType, OutputDataType>::LeakyReLU(
       const double alpha) : alpha(alpha)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void LeakyReLU<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     output = arma::max(input, alpha * input);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename DataType>
   void LeakyReLU<InputDataType, OutputDataType>::Backward(
       const DataType& input, const DataType& gy, DataType& g)
   {
     DataType derivative;
     derivative.set_size(arma::size(input));
     for (size_t i = 0; i < input.n_elem; ++i)
       derivative(i) = (input(i) >= 0) ? 1 : alpha;
   
     g = gy % derivative;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void LeakyReLU<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(alpha));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
