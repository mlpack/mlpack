
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_hard_tanh_impl.hpp:

Program Listing for File hard_tanh_impl.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_hard_tanh_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/hard_tanh_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_HARD_TANH_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_HARD_TANH_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "hard_tanh.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   HardTanH<InputDataType, OutputDataType>::HardTanH(
       const double maxValue,
       const double minValue) :
       maxValue(maxValue),
       minValue(minValue)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void HardTanH<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     output = input;
     for (size_t i = 0; i < input.n_elem; ++i)
     {
       output(i) = (output(i) > maxValue ? maxValue :
           (output(i) < minValue ? minValue : output(i)));
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename DataType>
   void HardTanH<InputDataType, OutputDataType>::Backward(
       const DataType& input, const DataType& gy, DataType& g)
   {
     g = gy;
     for (size_t i = 0; i < input.n_elem; ++i)
     {
       if (input(i) < minValue || input(i) > maxValue)
       {
         g(i) = 0;
       }
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void HardTanH<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(maxValue));
     ar(CEREAL_NVP(minValue));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
