
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_linear_impl.hpp:

Program Listing for File linear_impl.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_linear_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/linear_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_LINEAR_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "linear.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   Linear<InputDataType, OutputDataType, RegularizerType>::Linear() :
       inSize(0),
       outSize(0)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   Linear<InputDataType, OutputDataType, RegularizerType>::Linear(
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
   Linear<InputDataType, OutputDataType, RegularizerType>::Linear(
       const Linear& layer) :
       inSize(layer.inSize),
       outSize(layer.outSize),
       weights(layer.weights),
       regularizer(layer.regularizer)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   Linear<InputDataType, OutputDataType, RegularizerType>::Linear(
       Linear&& layer) :
       inSize(0),
       outSize(0),
       weights(std::move(layer.weights)),
       regularizer(std::move(layer.regularizer))
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   Linear<InputDataType, OutputDataType, RegularizerType>&
   Linear<InputDataType, OutputDataType, RegularizerType>::
   operator=(const Linear& layer)
   {
     if (this != &layer)
     {
       inSize = layer.inSize;
       outSize = layer.outSize;
       weights = layer.weights;
       regularizer = layer.regularizer;
     }
     return *this;
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   Linear<InputDataType, OutputDataType, RegularizerType>&
   Linear<InputDataType, OutputDataType, RegularizerType>::
   operator=(Linear&& layer)
   {
     if (this != &layer)
     {
       inSize = layer.inSize;
       outSize = layer.outSize;
       weights = std::move(layer.weights);
       regularizer = std::move(layer.regularizer);
     }
     return *this;
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   void Linear<InputDataType, OutputDataType, RegularizerType>::Reset()
   {
     weight = arma::mat(weights.memptr(), outSize, inSize, false, false);
     bias = arma::mat(weights.memptr() + weight.n_elem,
         outSize, 1, false, false);
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   template<typename eT>
   void Linear<InputDataType, OutputDataType, RegularizerType>::Forward(
       const arma::Mat<eT>& input, arma::Mat<eT>& output)
   {
     output = weight * input;
     output.each_col() += bias;
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   template<typename eT>
   void Linear<InputDataType, OutputDataType, RegularizerType>::Backward(
       const arma::Mat<eT>& /* input */, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
   {
     g = weight.t() * gy;
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   template<typename eT>
   void Linear<InputDataType, OutputDataType, RegularizerType>::Gradient(
       const arma::Mat<eT>& input,
       const arma::Mat<eT>& error,
       arma::Mat<eT>& gradient)
   {
     gradient.submat(0, 0, weight.n_elem - 1, 0) = arma::vectorise(
         error * input.t());
     gradient.submat(weight.n_elem, 0, gradient.n_elem - 1, 0) =
         arma::sum(error, 1);
     regularizer.Evaluate(weights, gradient);
   }
   
   template<typename InputDataType, typename OutputDataType,
       typename RegularizerType>
   template<typename Archive>
   void Linear<InputDataType, OutputDataType, RegularizerType>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(inSize));
     ar(CEREAL_NVP(outSize));
     ar(CEREAL_NVP(weights));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
