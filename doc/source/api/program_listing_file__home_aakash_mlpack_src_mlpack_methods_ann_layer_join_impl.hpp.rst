
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_join_impl.hpp:

Program Listing for File join_impl.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_join_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/join_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_JOIN_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_JOIN_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "join.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   Join<InputDataType, OutputDataType>::Join() :
       inSizeRows(0),
       inSizeCols(0)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void Join<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     inSizeRows = input.n_rows;
     inSizeCols = input.n_cols;
     output = arma::vectorise(input);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Join<InputDataType, OutputDataType>::Backward(
       const arma::Mat<eT>& /* input */,
       const arma::Mat<eT>& gy,
       arma::Mat<eT>& g)
   {
     g = arma::mat(((arma::Mat<eT>&) gy).memptr(), inSizeRows, inSizeCols, false,
         false);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void Join<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(inSizeRows));
     ar(CEREAL_NVP(inSizeCols));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
