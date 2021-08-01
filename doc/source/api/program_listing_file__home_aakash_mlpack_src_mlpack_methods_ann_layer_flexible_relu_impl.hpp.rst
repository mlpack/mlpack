
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_flexible_relu_impl.hpp:

Program Listing for File flexible_relu_impl.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_flexible_relu_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/flexible_relu_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_FLEXIBLERELU_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_FLEXIBLERELU_IMPL_HPP
   
   #include "flexible_relu.hpp"
   #include<algorithm>
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   FlexibleReLU<InputDataType, OutputDataType>::FlexibleReLU(
       const double alpha) : userAlpha(alpha)
   {
     this->alpha.set_size(1, 1);
     this->alpha(0) = userAlpha;
   }
   
   template<typename InputDataType, typename OutputDataType>
   void FlexibleReLU<InputDataType, OutputDataType>::Reset()
   {
     alpha(0) = userAlpha;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void FlexibleReLU<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     output = arma::clamp(input, 0.0, DBL_MAX) + alpha(0);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename DataType>
   void FlexibleReLU<InputDataType, OutputDataType>::Backward(
       const DataType& input, const DataType& gy, DataType& g)
   {
     g = gy % arma::clamp(arma::sign(input), 0.0, 1.0);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void FlexibleReLU<InputDataType, OutputDataType>::Gradient(
       const arma::Mat<eT>& input,
       const arma::Mat<eT>& error,
       arma::Mat<eT>& gradient)
   {
     if (gradient.n_elem == 0)
     {
       gradient.set_size(1, 1);
     }
   
     gradient(0) = arma::accu(error) / input.n_cols;
   }
   
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void FlexibleReLU<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version*/)
   {
     ar(CEREAL_NVP(alpha));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
