
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_log_softmax_impl.hpp:

Program Listing for File log_softmax_impl.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_log_softmax_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/log_softmax_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_LOG_SOFTMAX_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_LOG_SOFTMAX_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "log_softmax.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   LogSoftMax<InputDataType, OutputDataType>::LogSoftMax()
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void LogSoftMax<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     arma::mat maxInput = arma::repmat(arma::max(input), input.n_rows, 1);
     output = (maxInput - input);
   
     // Approximation of the base-e exponential function. The acuracy however is
     // about 0.00001 lower as using exp. Credits go to Leon Bottou.
     output.transform([](double x)
     {
       static constexpr double A0 = 1.0;
       static constexpr double A1 = 0.125;
       static constexpr double A2 = 0.0078125;
       static constexpr double A3 = 0.00032552083;
       static constexpr double A4 = 1.0172526e-5;
   
       if (x < 13.0)
       {
         double y = A0 + x * (A1 + x * (A2 + x * (A3 + x * A4)));
         y *= y;
         y *= y;
         y *= y;
         y = 1 / y;
   
         return y;
       }
   
       return 0.0;
     });
   
     maxInput.each_row() += arma::log(arma::sum(output));
     output = input - maxInput;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void LogSoftMax<InputDataType, OutputDataType>::Backward(
       const arma::Mat<eT>& input,
       const arma::Mat<eT>& gy,
       arma::Mat<eT>& g)
   {
     g = arma::exp(input) + gy;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void LogSoftMax<InputDataType, OutputDataType>::serialize(
       Archive& /* ar */,
       const uint32_t /* version */)
   {
     // Nothing to do here.
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
