
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_add_impl.hpp:

Program Listing for File add_impl.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_add_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/add_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_ADD_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_ADD_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "add.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   Add<InputDataType, OutputDataType>::Add(const size_t outSize) :
       outSize(outSize)
   {
     weights.set_size(WeightSize(), 1);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Add<InputDataType, OutputDataType>::Forward(
       const arma::Mat<eT>& input, arma::Mat<eT>& output)
   {
     output = input;
     output.each_col() += weights;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Add<InputDataType, OutputDataType>::Backward(
       const arma::Mat<eT>& /* input */,
       const arma::Mat<eT>& gy,
       arma::Mat<eT>& g)
   {
     g = gy;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Add<InputDataType, OutputDataType>::Gradient(
       const arma::Mat<eT>& /* input */,
       const arma::Mat<eT>& error,
       arma::Mat<eT>& gradient)
   {
     gradient = error;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void Add<InputDataType, OutputDataType>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(outSize));
   
     if (cereal::is_loading<Archive>())
       weights.set_size(outSize, 1);
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
