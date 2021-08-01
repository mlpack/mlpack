
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concat_performance_impl.hpp:

Program Listing for File concat_performance_impl.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concat_performance_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/concat_performance_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_PERFORMANCE_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_CONCAT_PERFORMANCE_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "concat_performance.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<
       typename OutputLayerType,
       typename InputDataType,
       typename OutputDataType
   >
   ConcatPerformance<
       OutputLayerType,
       InputDataType,
       OutputDataType
   >::ConcatPerformance(const size_t inSize, OutputLayerType&& outputLayer) :
       inSize(inSize),
       outputLayer(std::move(outputLayer))
   {
     // Nothing to do here.
   }
   
   template<
       typename OutputLayerType,
       typename InputDataType,
       typename OutputDataType
   >
   template<typename eT>
   double ConcatPerformance<
       OutputLayerType,
       InputDataType,
       OutputDataType
   >::Forward(const arma::Mat<eT>& input, arma::Mat<eT>& target)
   {
     const size_t elements = input.n_elem / inSize;
   
     double output = 0;
     for (size_t i = 0; i < input.n_elem; i+= elements)
     {
       arma::mat subInput = input.submat(i, 0, i + elements - 1, 0);
       output += outputLayer.Forward(subInput, target);
     }
   
     return output;
   }
   
   template<
       typename OutputLayerType,
       typename InputDataType,
       typename OutputDataType
   >
   template<typename eT>
   void ConcatPerformance<
       OutputLayerType,
       InputDataType,
       OutputDataType
   >::Backward(
       const arma::Mat<eT>& input,
       const arma::Mat<eT>& target,
       arma::Mat<eT>& output)
   {
     const size_t elements = input.n_elem / inSize;
   
     arma::mat subInput = input.submat(0, 0, elements - 1, 0);
     arma::mat subOutput;
   
     outputLayer.Backward(subInput, target, subOutput);
   
     output = arma::zeros(subOutput.n_elem, inSize);
     output.col(0) = subOutput;
   
     for (size_t i = elements, j = 0; i < input.n_elem; i+= elements, ++j)
     {
       subInput = input.submat(i, 0, i + elements - 1, 0);
       outputLayer.Backward(subInput, target, subOutput);
   
       output.col(j) = subOutput;
     }
   }
   
   template<
       typename OutputLayerType,
       typename InputDataType,
       typename OutputDataType
   >
   template<typename Archive>
   void ConcatPerformance<
       OutputLayerType,
       InputDataType,
       OutputDataType
   >::serialize(Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(inSize));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "concat_performance_impl.hpp"
   
   #endif
