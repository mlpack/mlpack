
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_lookup.hpp:

Program Listing for File lookup.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_lookup.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/lookup.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_LOOKUP_HPP
   #define MLPACK_METHODS_ANN_LAYER_LOOKUP_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   namespace mlpack {
   namespace ann /* Artificial Neural Network. */ {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class Lookup
   {
    public:
     Lookup(const size_t vocabSize = 0, const size_t embeddingSize = 0);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     template<typename eT>
     void Gradient(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& error,
                   arma::Mat<eT>& gradient);
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
     size_t VocabSize() const { return vocabSize; }
   
     size_t EmbeddingSize() const { return embeddingSize; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t vocabSize;
   
     size_t embeddingSize;
   
     OutputDataType weights;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     OutputDataType outputParameter;
   }; // class Lookup
   
   // Alias for using as embedding layer.
   template<typename MatType = arma::mat>
   using Embedding = Lookup<MatType, MatType>;
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "lookup_impl.hpp"
   
   #endif
