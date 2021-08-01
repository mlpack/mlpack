
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_softshrink_impl.hpp:

Program Listing for File softshrink_impl.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_softshrink_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/softshrink_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_SOFTSHRINK_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_SOFTSHRINK_IMPL_HPP
   
   // In case it hasn't yet been included
   #include "softshrink.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   // This constructor is called for Soft Shrink activation function.
   // lambda is a hyperparameter.
   template<typename InputDataType, typename OutputDataType>
   SoftShrink<InputDataType, OutputDataType>::SoftShrink(const double lambda) :
       lambda(lambda)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void SoftShrink<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     output = (input > lambda) % (input - lambda) + (
       input < -lambda) % (input + lambda);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename DataType>
   void SoftShrink<InputDataType, OutputDataType>::Backward(
       const DataType& input, DataType& gy, DataType& g)
   {
     DataType derivative;
     derivative = (arma::ones(arma::size(input)) - (input == 0));
     g = gy % derivative;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void SoftShrink<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(lambda));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
