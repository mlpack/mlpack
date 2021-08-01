
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_parametric_relu_impl.hpp:

Program Listing for File parametric_relu_impl.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_parametric_relu_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/parametric_relu_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_PReLU_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_PReLU_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "parametric_relu.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   PReLU<InputDataType, OutputDataType>::PReLU(
       const double userAlpha) : userAlpha(userAlpha)
   {
     alpha.set_size(WeightSize(), 1);
     alpha(0) = userAlpha;
   }
   
   template<typename InputDataType, typename OutputDataType>
   void PReLU<InputDataType, OutputDataType>::Reset()
   {
     alpha(0) = userAlpha;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void PReLU<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     output = input;
     arma::uvec negative = arma::find(input < 0);
     output(negative) = input(negative) * alpha(0);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename DataType>
   void PReLU<InputDataType, OutputDataType>::Backward(
       const DataType& input, const DataType& gy, DataType& g)
   {
     DataType derivative;
     derivative.set_size(arma::size(input));
     for (size_t i = 0; i < input.n_elem; ++i)
     {
       derivative(i) = (input(i) >= 0) ? 1 : alpha(0);
     }
   
     g = gy % derivative;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void PReLU<InputDataType, OutputDataType>::Gradient(
       const arma::Mat<eT>& input,
       const arma::Mat<eT>& error,
       arma::Mat<eT>& gradient)
   {
     if (gradient.n_elem == 0)
     {
       gradient = arma::zeros<arma::mat>(1, 1);
     }
   
     arma::mat zeros = arma::zeros<arma::mat>(input.n_rows, input.n_cols);
     gradient(0) = arma::accu(error % arma::min(zeros, input)) / input.n_cols;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void PReLU<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(alpha));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
