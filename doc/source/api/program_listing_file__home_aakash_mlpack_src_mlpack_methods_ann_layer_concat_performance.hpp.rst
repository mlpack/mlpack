
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concat_performance.hpp:

Program Listing for File concat_performance.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concat_performance.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/concat_performance.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_PERFORMANCE_HPP
   #define MLPACK_METHODS_ANN_LAYER_CONCAT_PERFORMANCE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "layer_types.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename OutputLayerType = NegativeLogLikelihood<>,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class ConcatPerformance
   {
    public:
     ConcatPerformance(const size_t inSize = 0,
                       OutputLayerType&& outputLayer = OutputLayerType());
   
     /*
      * Computes the Negative log likelihood.
      *
      * @param input Input data used for evaluating the specified function.
      * @param output Resulting output activation.
      */
     template<typename eT>
     double Forward(const arma::Mat<eT>& input, arma::Mat<eT>& target);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& target,
                   arma::Mat<eT>& output);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     size_t InSize() const { return inSize; }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */);
   
    private:
     size_t inSize;
   
     OutputLayerType outputLayer;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   }; // class ConcatPerformance
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "concat_performance_impl.hpp"
   
   #endif
