
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multiply_constant_impl.hpp:

Program Listing for File multiply_constant_impl.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multiply_constant_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/multiply_constant_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_MULTIPLY_CONSTANT_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_MULTIPLY_CONSTANT_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "multiply_constant.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   MultiplyConstant<InputDataType, OutputDataType>::MultiplyConstant(
       const double scalar) : scalar(scalar)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   MultiplyConstant<InputDataType, OutputDataType>::MultiplyConstant(
       const MultiplyConstant& layer) :
       scalar(layer.scalar)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   MultiplyConstant<InputDataType, OutputDataType>::MultiplyConstant(
       MultiplyConstant&& layer) :
       scalar(std::move(layer.scalar))
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   MultiplyConstant<InputDataType, OutputDataType>&
   MultiplyConstant<InputDataType, OutputDataType>::operator=(
       const MultiplyConstant& layer)
   {
     if (this != &layer)
     {
       scalar = layer.scalar;
     }
     return *this;
   }
   
   template<typename InputDataType, typename OutputDataType>
   MultiplyConstant<InputDataType, OutputDataType>&
   MultiplyConstant<InputDataType, OutputDataType>::operator=(
       MultiplyConstant&& layer)
   {
     if (this != &layer)
     {
       scalar = std::move(layer.scalar);
     }
     return *this;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void MultiplyConstant<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     output = input * scalar;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename DataType>
   void MultiplyConstant<InputDataType, OutputDataType>::Backward(
       const DataType& /* input */, const DataType& gy, DataType& g)
   {
     g = gy * scalar;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void MultiplyConstant<InputDataType, OutputDataType>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(scalar));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
