
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_select_impl.hpp:

Program Listing for File select_impl.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_select_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/select_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_SELECT_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_SELECT_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "constant.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   Select<InputDataType, OutputDataType>::Select(
       const size_t index,
       const size_t elements) :
       index(index),
       elements(elements)
     {
       // Nothing to do here.
     }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Select<InputDataType, OutputDataType>::Forward(
       const arma::Mat<eT>& input, arma::Mat<eT>& output)
   {
     if (elements == 0)
     {
       output = input.col(index);
     }
     else
     {
       output = input.submat(0, index, elements - 1, index);
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Select<InputDataType, OutputDataType>::Backward(
       const arma::Mat<eT>& /* input */,
       const arma::Mat<eT>& gy,
       arma::Mat<eT>& g)
   {
     if (elements == 0)
     {
       g = gy;
     }
     else
     {
       g = gy.submat(0, 0, elements - 1, 0);
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void Select<InputDataType, OutputDataType>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(index));
     ar(CEREAL_NVP(elements));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
