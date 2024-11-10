/**
 * @file core/metrics/bleu_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of BLEU class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_BLEU_IMPL_HPP
#define MLPACK_CORE_METRICS_BLEU_IMPL_HPP

// In case it hasn't been included.
#include "bleu.hpp"

namespace mlpack {

template <typename ElemType, typename PrecisionType>
BLEU<ElemType, PrecisionType>::BLEU(const size_t maxOrder) :
    maxOrder(maxOrder),
    translationLength(0),
    referenceLength(0)
{
  // Nothing to do here.
}

template <typename ElemType, typename PrecisionType>
template <typename WordVector>
std::map<WordVector, size_t> BLEU<ElemType, PrecisionType>::GetNGrams(
    const WordVector& segment)
{
  std::map<WordVector, size_t> ngramsCount;
  for (size_t order = 1; order < maxOrder + 1; ++order)
  {
    for (size_t i = 0; i + order < segment.size() + 1; ++i)
    {
      WordVector seq = WordVector(segment.cbegin() + i,
                                  segment.cbegin() + i + order);
      ngramsCount[seq]++;
    }
  }
  return ngramsCount;
}

template <typename ElemType, typename PrecisionType>
template <typename ReferenceCorpusType, typename TranslationCorpusType>
ElemType BLEU<ElemType, PrecisionType>::Evaluate(
    const ReferenceCorpusType& referenceCorpus,
    const TranslationCorpusType& translationCorpus,
    const bool smooth)
{
  // WordVector is a string container type.
  // Also, TranslationCorpusType is an array of such containers.
  using WordVector = typename TranslationCorpusType::value_type;

  // matchesByOrder: It catches how many times sequence of a particular order
  // is encountered in both reference corpus and translation corpus.
  std::vector<size_t> matchesByOrder(maxOrder, 0);

  // possibleMatchesByOrder: It tracks how many possible matches can be in the
  // translation corpus.
  std::vector<size_t> possibleMatchesByOrder(maxOrder, 0);

  // referenceLength: It is the sum of minimum length of the paragraph from
  // various documents.
  // translationLength: It is the sum of length of each paragraphs.
  referenceLength = 0, translationLength = 0;

  auto refIt = referenceCorpus.cbegin();
  auto trIt = translationCorpus.cbegin();
  for (; refIt != referenceCorpus.cend() && trIt != translationCorpus.cend();
      ++refIt, ++trIt)
  {
    size_t min = std::numeric_limits<size_t>::max();
    for (const auto& t : *refIt)
    {
      if (min > t.size())
      {
        min = t.size();
      }
    }

    if (min == std::numeric_limits<size_t>::max())
      min = 0;

    referenceLength += min;
    translationLength += trIt->size();

    // mergedRefNGramCounts: It accumulates all the similar n-grams from
    // various references or documents, so that there is no repetition of
    // any key (sequence of order n).
    std::map<WordVector, size_t> mergedRefNGramCounts;
    for (const auto& t : *refIt)
    {
      // ngram: It holds the n-grams of each document/reference.
      const std::map<WordVector, size_t> ngrams = GetNGrams(t);
      for (auto it = ngrams.cbegin(); it != ngrams.cend(); ++it)
      {
        mergedRefNGramCounts[it->first] = std::max(it->second,
            mergedRefNGramCounts[it->first]);
      }
    }
    // translationNGramCounts: It extracts the n-grams of the generated text
    // sequence.
    const std::map<WordVector, size_t> translationNGramCounts
        = GetNGrams(*trIt);

    // overlap: It holds those keys (sequence of order n) which are common to
    // reference corpus and translation corpus.
    std::map<WordVector, size_t> overlap;
    for (auto it = translationNGramCounts.cbegin();
         it != translationNGramCounts.cend();
         ++it)
    {
      auto mergedIt = mergedRefNGramCounts.find(it->first);
      if (mergedIt != mergedRefNGramCounts.end())
      {
        // If the key (sequence of order n) is present in both translation
        // corpus as well as reference corpus, then the minimum number of
        // counts it has occurred in any is considered.
        overlap[it->first] = std::min(mergedIt->second, it->second);
      }
    }

    for (auto it = overlap.cbegin(); it != overlap.cend(); ++it)
    {
      matchesByOrder[it->first.size() - 1] += it->second;
    }

    for (size_t order = 1; order < maxOrder + 1; ++order)
    {
      if (order < trIt->size() + 1)
        possibleMatchesByOrder[order - 1] += trIt->size() - order + 1;
    }
  }

  precisions = PrecisionType(maxOrder, 0.0);

  if (smooth)
  {
    for (size_t i = 0; i < maxOrder; ++i)
    {
      precisions[i]
          = (matchesByOrder[i] + 1.0) / (possibleMatchesByOrder[i] + 1.0);
    }
  }
  else
  {
    for (size_t i = 0; i < maxOrder; ++i)
    {
      if (possibleMatchesByOrder[i] > 0)
        precisions[i] = ElemType(matchesByOrder[i]) / possibleMatchesByOrder[i];
      else
        precisions[i] = 0.0;
    }
  }

  ElemType minPrecision = std::numeric_limits<ElemType>::max();
  for (size_t i = 0; i < maxOrder; ++i)
  {
    if (minPrecision > precisions[i])
      minPrecision = precisions[i];
  }

  ElemType geometricMean;
  if (minPrecision > 0)
  {
    ElemType pLogSum = 0.0;
    for (const auto& t : precisions)
    {
      pLogSum += (1.0 / maxOrder) * std::log(t);
    }
    geometricMean = std::exp(pLogSum);
  }
  else
    geometricMean = 0.0;

  ratio = ElemType(translationLength);
  if (referenceLength > 0)
    ratio /= referenceLength;

  brevityPenalty = (ratio > 1.0) ? 1.0 : std::exp(1.0 - 1.0 / ratio);
  bleuScore = geometricMean * brevityPenalty;

  return bleuScore;
}

template <typename ElemType, typename PrecisionType>
template <typename Archive>
void BLEU<ElemType, PrecisionType>::serialize(Archive& ar,
    const uint32_t version)
{
  ar(CEREAL_NVP(maxOrder));
}

} // namespace mlpack

#endif
