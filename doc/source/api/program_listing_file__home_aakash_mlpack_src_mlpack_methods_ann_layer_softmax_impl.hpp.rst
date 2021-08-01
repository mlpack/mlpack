
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_softmax_impl.hpp:

Program Listing for File softmax_impl.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_softmax_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/softmax_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_SOFTMAX_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_SOFTMAX_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "softmax.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   Softmax<InputDataType, OutputDataType>::Softmax()
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void Softmax<InputDataType, OutputDataType>::Forward(
       const InputType& input,
       OutputType& output)
   {
     InputType softmaxInput = arma::exp(input.each_row() -
         arma::max(input, 0));
     output = softmaxInput.each_row() / sum(softmaxInput, 0);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Softmax<InputDataType, OutputDataType>::Backward(
       const arma::Mat<eT>& input,
       const arma::Mat<eT>& gy,
       arma::Mat<eT>& g)
   {
     g = input % (gy - arma::repmat(arma::sum(gy % input), input.n_rows, 1));
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void Softmax<InputDataType, OutputDataType>::serialize(
       Archive& /* ar */,
       const uint32_t /* version */)
   {
     // Nothing to do here.
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
