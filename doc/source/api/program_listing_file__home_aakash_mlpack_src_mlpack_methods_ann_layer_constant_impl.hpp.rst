
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_constant_impl.hpp:

Program Listing for File constant_impl.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_constant_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/constant_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_CONSTANT_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_CONSTANT_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "constant.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   Constant<InputDataType, OutputDataType>::Constant(
       const size_t outSize,
       const double scalar) :
       inSize(0),
       outSize(outSize)
   {
     constantOutput = OutputDataType(outSize, 1);
     constantOutput.fill(scalar);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void Constant<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     if (inSize == 0)
     {
       inSize = input.n_elem;
     }
   
     output = constantOutput;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename DataType>
   void Constant<InputDataType, OutputDataType>::Backward(
       const DataType& /* input */, const DataType& /* gy */, DataType& g)
   {
     g = arma::zeros<DataType>(inSize, 1);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void Constant<InputDataType, OutputDataType>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(constantOutput));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
