
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_metrics_bleu.hpp:

Program Listing for File bleu.hpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_metrics_bleu.hpp>` (``/home/aakash/mlpack/src/mlpack/core/metrics/bleu.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_METRICS_BLEU_HPP
   #define MLPACK_CORE_METRICS_BLEU_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace metric {
   
   template <typename ElemType = float,
             typename PrecisionType = std::vector<ElemType>
   >
   class BLEU
   {
    public:
     BLEU(const size_t maxOrder = 4);
   
     template <typename ReferenceCorpusType, typename TranslationCorpusType>
     ElemType Evaluate(const ReferenceCorpusType& referenceCorpus,
                       const TranslationCorpusType& translationCorpus,
                       const bool smooth = false);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
     size_t MaxOrder() const { return maxOrder; }
     size_t& MaxOrder() { return maxOrder; }
   
     ElemType BLEUScore() const { return bleuScore; }
   
     ElemType BrevityPenalty() const { return brevityPenalty; }
   
     size_t TranslationLength() const { return translationLength; }
   
     size_t ReferenceLength() const { return referenceLength; }
   
     ElemType Ratio() const { return ratio; }
   
     PrecisionType const& Precisions() const { return precisions; }
   
    private:
     template <typename WordVector>
     std::map<WordVector, size_t> GetNGrams(const WordVector& segment);
   
     size_t maxOrder;
   
     ElemType bleuScore;
   
     ElemType brevityPenalty;
   
     size_t translationLength;
   
     size_t referenceLength;
   
     ElemType ratio;
   
     PrecisionType precisions;
   };
   
   } // namespace metric
   } // namespace mlpack
   
   // Include implementation.
   #include "bleu_impl.hpp"
   
   #endif
