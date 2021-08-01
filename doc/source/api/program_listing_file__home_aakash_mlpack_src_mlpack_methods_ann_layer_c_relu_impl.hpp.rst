
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_c_relu_impl.hpp:

Program Listing for File c_relu_impl.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_c_relu_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/c_relu_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_C_RELU_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_C_RELU_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "c_relu.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   CReLU<InputDataType, OutputDataType>::CReLU()
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void CReLU<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     output = arma::join_cols(arma::max(input, 0.0 * input), arma::max(
         (-1 * input), 0.0 * input));
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename DataType>
   void CReLU<InputDataType, OutputDataType>::Backward(
       const DataType& input, const DataType& gy, DataType& g)
   {
     DataType temp;
     temp = gy % (input >= 0.0);
     g = temp.rows(0, (input.n_rows / 2 - 1)) - temp.rows(input.n_rows / 2,
         (input.n_rows - 1));
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void CReLU<InputDataType, OutputDataType>::serialize(
       Archive& /* ar */,
       const uint32_t /* version */)
   {
     // Nothing to do here.
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
