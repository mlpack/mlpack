
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_log_softmax.hpp:

Program Listing for File log_softmax.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_log_softmax.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/log_softmax.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_LOG_SOFTMAX_HPP
   #define MLPACK_METHODS_ANN_LAYER_LOG_SOFTMAX_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class LogSoftMax
   {
    public:
     LogSoftMax();
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     InputDataType& Delta() const { return delta; }
     InputDataType& Delta() { return delta; }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   }; // class LogSoftmax
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "log_softmax_impl.hpp"
   
   #endif
