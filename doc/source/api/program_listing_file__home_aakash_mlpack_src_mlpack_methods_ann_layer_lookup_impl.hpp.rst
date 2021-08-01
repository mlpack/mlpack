
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_lookup_impl.hpp:

Program Listing for File lookup_impl.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_lookup_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/lookup_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_LOOKUP_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_LOOKUP_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "lookup.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <typename InputDataType, typename OutputDataType>
   Lookup<InputDataType, OutputDataType>::Lookup(
       const size_t vocabSize,
       const size_t embeddingSize) :
       vocabSize(vocabSize),
       embeddingSize(embeddingSize)
   {
     weights.set_size(embeddingSize, vocabSize);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Lookup<InputDataType, OutputDataType>::Forward(
       const arma::Mat<eT>& input, arma::Mat<eT>& output)
   {
     const size_t seqLength = input.n_rows;
     const size_t batchSize = input.n_cols;
   
     output.set_size(embeddingSize * seqLength, batchSize);
   
     for (size_t i = 0; i < batchSize; ++i)
     {
       // ith column of output is a vectorized form of a matrix of shape
       // (embeddingSize, seqLength) selected as a combination of columns from the
       // weights.
       output.col(i) = arma::vectorise(weights.cols(
           arma::conv_to<arma::uvec>::from(input.col(i)) - 1));
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Lookup<InputDataType, OutputDataType>::Backward(
       const arma::Mat<eT>& /* input */,
       const arma::Mat<eT>& /* gy */,
       arma::Mat<eT>& /* g */)
   {
     Log::Fatal << "Lookup cannot be used as an intermediate layer." << std::endl;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Lookup<InputDataType, OutputDataType>::Gradient(
       const arma::Mat<eT>& input,
       const arma::Mat<eT>& error,
       arma::Mat<eT>& gradient)
   {
     const size_t seqLength = input.n_rows;
     const size_t batchSize = input.n_cols;
   
     arma::Cube<eT> errorTemp(const_cast<arma::Mat<eT>&>(error).memptr(),
         embeddingSize, seqLength, batchSize, false, false);
   
     gradient.set_size(arma::size(weights));
     gradient.zeros();
   
     for (size_t i = 0; i < batchSize; ++i)
     {
       gradient.cols(arma::conv_to<arma::uvec>::from(input.col(i)) - 1)
           += errorTemp.slice(i);
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void Lookup<InputDataType, OutputDataType>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(vocabSize));
     ar(CEREAL_NVP(embeddingSize));
   
     // This is inefficient, but we have to allocate this memory so that
     // WeightSetVisitor gets the right size.
     if (cereal::is_loading<Archive>())
       weights.set_size(embeddingSize, vocabSize);
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
