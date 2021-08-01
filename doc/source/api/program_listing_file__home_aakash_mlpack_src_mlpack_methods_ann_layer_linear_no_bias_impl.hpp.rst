
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_linear_no_bias_impl.hpp:

Program Listing for File linear_no_bias_impl.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_linear_no_bias_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/linear_no_bias_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_NO_BIAS_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_LINEAR_NO_BIAS_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "linear_no_bias.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   LinearNoBias<InputDataType, OutputDataType, RegularizerType>::LinearNoBias() :
       inSize(0),
       outSize(0)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   LinearNoBias<InputDataType, OutputDataType, RegularizerType>::LinearNoBias(
       const size_t inSize,
       const size_t outSize,
       RegularizerType regularizer) :
       inSize(inSize),
       outSize(outSize),
       regularizer(regularizer)
   {
     weights.set_size(WeightSize(), 1);
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   void LinearNoBias<InputDataType, OutputDataType, RegularizerType>::Reset()
   {
     weight = arma::mat(weights.memptr(), outSize, inSize, false, false);
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   template<typename eT>
   void LinearNoBias<InputDataType, OutputDataType, RegularizerType>::Forward(
       const arma::Mat<eT>& input, arma::Mat<eT>& output)
   {
     output = weight * input;
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   template<typename eT>
   void LinearNoBias<InputDataType, OutputDataType, RegularizerType>::Backward(
       const arma::Mat<eT>& /* input */, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
   {
     g = weight.t() * gy;
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   template<typename eT>
   void LinearNoBias<InputDataType, OutputDataType, RegularizerType>::Gradient(
       const arma::Mat<eT>& input,
       const arma::Mat<eT>& error,
       arma::Mat<eT>& gradient)
   {
     gradient.submat(0, 0, weight.n_elem - 1, 0) = arma::vectorise(
         error * input.t());
     regularizer.Evaluate(weights, gradient);
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   template<typename Archive>
   void LinearNoBias<InputDataType, OutputDataType, RegularizerType>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(inSize));
     ar(CEREAL_NVP(outSize));
   
     // This is inefficient, but necessary so that WeightSetVisitor sets the right
     // size.
     if (cereal::is_loading<Archive>())
       weights.set_size(outSize * inSize, 1);
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
