/**
 * @file core/metrics/bleu.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition of BLEU class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_BLEU_HPP
#define MLPACK_CORE_METRICS_BLEU_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * BLEU, or the Bilingual Evaluation Understudy, is an algorithm for evaluating
 * the quality of text which has been machine translated from one natural
 * language to another. It can also be used to evaluate text generated for a
 * suite of natural language processing tasks.
 *
 * The BLEU score is calculated using the following formula:
 *
 * \f{eqnarray*}{
 * \text{B} &=& bp \cdot \exp \left(\sum_{n=1}^{N} w \log p_n \right) \\
 * \text{where,} \\
 * bp &=& \text{brevity penalty} =
 * \begin{cases}
 *   1 & \text{if ratio} > 1 \\
 *   \exp \left(1-\frac{1}{ratio}\right) & \text{otherwise}
 * \end{cases} \\
 * p_n &=& \text{modified precision for n-gram,} \\
 * w &=& \frac {1}{maxOrder}, \\
 * ratio &=& \text{translation to reference length ratio,} \\
 * maxOrder &=& \text{maximum length of tokens in n-grams.}
 * \f}
 *
 * The value of BLEU Score lies in between 0 and 1.
 *
 * @tparam ElemType Type of the quantities in BLEU, e.g. (long double,
 *         double, float).
 * @tparam PrecisionType Container type for precision for corresponding order.
 *         e.g. (std::vector<float>, std::vector<double>, or any such boost or
 *         armadillo container).
 */
template <typename ElemType = float,
          typename PrecisionType = std::vector<ElemType>
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
   * @param referenceCorpus It is an array of various references or documents.
   * So, the \f$ referenceCorpus = \{reference_1, reference_2, \ldots \} \f$
   * and each reference is an array of paragraphs. So,
   * \f$ reference_i = \{paragraph_1, paragraph_2, \ldots \} \f$
   * and then each paragraph is an array of tokenized words/string. Like,
   * \f$ paragraph_i = \{word_1, word_2, \ldots \} \f$.
   * For ex.
   * ```
   * refCorpus = {{{"this", "is", "paragraph", "1", "from", "document", "1"},
   *               {"this", "is", "paragraph", "2", "from", "document", "1"}},
   *
   *              {{"this", "is", "paragraph", "1", "from", "document", "2"},
   *               {"this", "is", "paragraph", "2", "from", "document", "2"}}}
   * ```
   * @param translationCorpus It is an array of paragraphs which has been
   * machine translated or generated for any natural language processing task.
   * Like, \f$ translationCorpus = \{paragraph_1, paragraph_2, \ldots \} \f$.
   * And then, each paragraph is an array of words. The ith paragraph from the
   * corpus is \f$ paragraph_i = \{word_1, word_2, \ldots \} \f$.
   * For ex.
   * ```
   * transCorpus = {{"this", "is", "generated", "paragraph", "1"},
   *                {"this", "is", "generated", "paragraph", "2"}}
   * ```
   * @param smooth Whether or not to apply Lin et al. 2004 smoothing.
   * @return The Evaluate method returns the BLEU Score. This method also
   * calculates other BLEU metrics (brevity penalty, translation length, reference
   * length, ratio and precisions) which can be accessed by their corresponding
   * accessor methods.
   */
  template <typename ReferenceCorpusType, typename TranslationCorpusType>
  ElemType Evaluate(const ReferenceCorpusType& referenceCorpus,
                    const TranslationCorpusType& translationCorpus,
                    const bool smooth = false);

  //! Serialize the metric.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

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
  template <typename WordVector>
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

} // namespace mlpack

// Include implementation.
#include "bleu_impl.hpp"

#endif
