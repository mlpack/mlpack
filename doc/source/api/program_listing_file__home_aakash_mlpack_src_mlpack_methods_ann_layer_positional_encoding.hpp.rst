
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_positional_encoding.hpp:

Program Listing for File positional_encoding.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_positional_encoding.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/positional_encoding.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_POSITIONAL_ENCODING_HPP
   #define MLPACK_METHODS_ANN_LAYER_POSITIONAL_ENCODING_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class PositionalEncoding
   {
    public:
     PositionalEncoding();
   
     PositionalEncoding(const size_t embedDim,
                        const size_t maxSequenceLength);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     InputDataType const& InputParameter() const { return inputParameter; }
     InputDataType& InputParameter() { return inputParameter; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     InputDataType const& Encoding() const { return positionalEncoding; }
   
     size_t InputShape() const
     {
       return embedDim * maxSequenceLength;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     void InitPositionalEncoding();
   
     size_t embedDim;
   
     size_t maxSequenceLength;
   
     InputDataType positionalEncoding;
   
     OutputDataType delta;
   
     InputDataType inputParameter;
   
     OutputDataType outputParameter;
   }; // class PositionalEncoding
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "positional_encoding_impl.hpp"
   
   #endif
