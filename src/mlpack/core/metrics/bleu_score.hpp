/**
 * @file core/metrics/bleu_score.hpp
 * @author Mrityunjay Tripathi
 *
 * BLEU, or the Bilingual Evaluation Understudy, is an algorithm for evaluating
 * the quality of text which has been machine translated from one natural
 * language to another. It can also be used to evaluate text generated for a
 * suite of natural language processing tasks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_BLEU_SCORE_HPP
#define MLPACK_CORE_METRICS_BLEU_SCORE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace metric {

/**
 * @tparam ElemType Type of the quantities in BLEU, eg. (long double,
 *         double, float)
 * @tparam PrecisionType Container type for precision for corresponding order.
 */
template <typename ElemType = float,
          typename PrecisionType = std::vector<float>
>
class BLEU
{
 public:
  /**
   * Create an instance of BLEU class.
   *
   * @param maxOrder The maximum length of tokens in n-grams.
   */
  BLEU(const size_t maxOrder = 4);

  /**
   * Computes the BLEU Score.
   *
   * @tparam ReferenceCorpusType Type of reference corpus.
   * @tparam TranslationCorpusType Type of translation corpus.
   * @param referenceCorpus Reference corpus.
   * @param translationCorpus Translation corpus.
   * @param smooth Whether or not to apply Lin et al. 2004 smoothing.
   */
  template <typename ReferenceCorpusType, typename TranslationCorpusType>
  ElemType Evaluate(const ReferenceCorpusType& referenceCorpus,
                    const TranslationCorpusType& translationCorpus,
                    const bool smooth = false);

  //! Serialize the metric (nothing to do).
  template<typename Archive>
  void serialize(Archive& /* ar */, const unsigned int /* version */) { }

  //! Get the value of maximum length of tokens in n-grams.
  size_t MaxOrder() const { return maxOrder; }
  //! Modify the value of maximum length of tokens in n-grams.
  size_t& MaxOrder() { return maxOrder; }

  //! Get the BLEU Score.
  ElemType BLEUScore() const { return bleuScore; }

  //! Get the brevity penalty.
  ElemType BrevityPenalty() const { return brevityPenalty; }

  //! Get the value of translation length.
  size_t TranslationLength() const { return translationLength; }

  //! Get the value of reference length.
  size_t ReferenceLength() const { return referenceLength; }

  //! Get the ratio of translation to reference length ratio.
  ElemType Ratio() const { return ratio; }

  //! Get the precisions for corresponding order.
  PrecisionType const& Precisions() const { return precisions; }

 private:
  /**
   * Extracts all the n-grams.
   *
   * @tparam WordVector Type of the tokenized vector.
   * @param segment Tokenized sequence represented in form of vector.
   */
  template <typename WordVector = std::vector<std::string>>
  std::map<WordVector, size_t> GetNGrams(const WordVector& segment);

  //! Locally-stored value of maximum length of tokens in n-grams.
  size_t maxOrder;

  //! Locally-stored BLEU score.
  ElemType bleuScore;

  //! Locally-stored brevity penalty. It is a penalty for short machine
  //! translation.
  ElemType brevityPenalty;

  //! Locally-stored translation length.
  size_t translationLength;

  //! Locally-stored reference length.
  size_t referenceLength;

  //! Locally-stored translation to reference length ratio.
  ElemType ratio;

  //! Locally stored precision for corresponding order.
  PrecisionType precisions;
};

} // namespace metric
} // namespace mlpack

// Include implementation.
#include "bleu_score_impl.hpp"

#endif
