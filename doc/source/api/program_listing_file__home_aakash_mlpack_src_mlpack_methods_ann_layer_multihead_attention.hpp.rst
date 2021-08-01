
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multihead_attention.hpp:

Program Listing for File multihead_attention.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multihead_attention.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/multihead_attention.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     author  = {Ashish Vaswani, Llion Jones, Noam Shazeer, Niki Parmar,
                Aidan N. Gomez, Jakob Uszkoreit, Łukasz Kaiser,
                Illia Polosukhin},
     title   = {Attention Is All You Need},
     year    = {2017},
     url     = {http://arxiv.org/abs/1706.03762v5}
   }
   
   #ifndef MLPACK_METHODS_ANN_LAYER_MULTIHEAD_ATTENTION_HPP
   #define MLPACK_METHODS_ANN_LAYER_MULTIHEAD_ATTENTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/layer/softmax.hpp>
   #include <mlpack/methods/ann/layer/dropout.hpp>
   #include <mlpack/methods/ann/init_rules/glorot_init.hpp>
   #include <mlpack/methods/ann/regularizer/no_regularizer.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat,
       typename RegularizerType = NoRegularizer
   >
   class MultiheadAttention
   {
    public:
     MultiheadAttention();
   
     MultiheadAttention(const size_t tgtSeqLen,
                        const size_t srcSeqLen,
                        const size_t embedDim,
                        const size_t numHeads);
   
     void Reset();
   
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
   
     size_t WeightSize() const { return 4 * (embedDim + 1) * embedDim; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     size_t TgtSeqLen() const { return tgtSeqLen; }
     size_t& TgtSeqLen() { return tgtSeqLen; }
   
     size_t SrcSeqLen() const { return srcSeqLen; }
     size_t& SrcSeqLen() { return srcSeqLen; }
   
     size_t EmbedDim() const { return embedDim; }
     size_t& EmbedDim() { return embedDim; }
   
     size_t NumHeads() const { return numHeads; }
     size_t& NumHeads() { return numHeads; }
   
     OutputDataType const& AttentionMask() const { return attnMask; }
     OutputDataType& AttentionMask() { return attnMask; }
   
     OutputDataType const& KeyPaddingMask() const { return keyPaddingMask; }
     OutputDataType& KeyPaddingMask() { return keyPaddingMask; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return grad; }
     OutputDataType& Gradient() { return grad; }
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     size_t InputShape() const
     {
       return embedDim * (tgtSeqLen + 2 * srcSeqLen);
     }
   
    private:
     typedef typename OutputDataType::elem_type ElemType;
   
     size_t tgtSeqLen;
   
     size_t srcSeqLen;
   
     size_t embedDim;
   
     size_t numHeads;
   
     size_t headDim;
   
     OutputDataType attnMask;
   
     OutputDataType keyPaddingMask;
   
     OutputDataType queryWt;
   
     OutputDataType keyWt;
   
     OutputDataType valueWt;
   
     OutputDataType outWt;
   
     OutputDataType qBias;
   
     OutputDataType kBias;
   
     OutputDataType vBias;
   
     OutputDataType outBias;
   
     OutputDataType weights;
   
     arma::Cube<ElemType> qProj;
   
     arma::Cube<ElemType> kProj;
   
     arma::Cube<ElemType> vProj;
   
     arma::Cube<ElemType> scores;
   
     arma::Cube<ElemType> attnOut;
   
     Softmax<InputDataType, OutputDataType> softmax;
   
     OutputDataType delta;
   
     OutputDataType grad;
   
     OutputDataType outputParameter;
   
     RegularizerType regularizer;
   }; // class MultiheadAttention
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "multihead_attention_impl.hpp"
   
   #endif
