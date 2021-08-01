
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_softmin_impl.hpp:

Program Listing for File softmin_impl.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_softmin_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/softmin_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_SOFTMIN_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_SOFTMIN_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "softmin.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   Softmin<InputDataType, OutputDataType>::Softmin()
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void Softmin<InputDataType, OutputDataType>::Forward(
       const InputType& input,
       OutputType& output)
   {
     InputType softminInput = arma::exp(-(input.each_row() -
         arma::min(input, 0)));
     output = softminInput.each_row() / sum(softminInput, 0);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Softmin<InputDataType, OutputDataType>::Backward(
       const arma::Mat<eT>& input,
       const arma::Mat<eT>& gy,
       arma::Mat<eT>& g)
   {
     g = input % (gy - arma::repmat(arma::sum(gy % input), input.n_rows, 1));
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void Softmin<InputDataType, OutputDataType>::serialize(
       Archive& /* ar */,
       const uint32_t /* version */)
   {
     // Nothing to do here.
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
