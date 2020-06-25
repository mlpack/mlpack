/**
 * @file core/metrics/bleu_score_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of BLEUScore class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_BLEU_SCORE_IMPLHPP
#define MLPACK_CORE_METRICS_BLEU_SCORE_IMPLHPP

// In case it hasn't been included.
#include "bleu_score.hpp"

namespace mlpack {
namespace metric {

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
    for (size_t i = 0; i < segment.size() - order + 1; ++i)
    {
      WordVector seq = WordVector(segment.begin() + i,
                                  segment.begin() + i + order);
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
  typedef typename TranslationCorpusType::value_type WordVector;
  std::vector<size_t> matchesByOrder(maxOrder, 0);
  std::vector<size_t> possibleMatchesByOrder(maxOrder, 0);
  referenceLength = 0, translationLength = 0;

  auto refIt = referenceCorpus.cbegin();
  auto trIt = translationCorpus.cbegin();
  for (; refIt != referenceCorpus.cend(), trIt != translationCorpus.cend();
      ++refIt, ++trIt)
  {
    size_t min = std::numeric_limits<size_t>::max();
    for (auto t: *refIt)
    {
      if (min > t.size())
      {
        min = t.size();
      }
    }
    referenceLength += min;
    translationLength += trIt->size();

    std::map<WordVector, size_t> mergedRefNGramCounts;
    for (auto t: *refIt)
    {
      const std::map<WordVector, size_t> ngrams = GetNGrams(t);
      for (auto it = ngrams.cbegin(); it != ngrams.cend(); ++it)
      {
        if (!mergedRefNGramCounts[it->first])
          mergedRefNGramCounts[it->first] = it->second;
        else
          mergedRefNGramCounts[it->first]
              = std::max(mergedRefNGramCounts[it->first], it->second);
      }
    }

    std::map<WordVector, size_t> translationNGramCounts = GetNGrams(*trIt);
    std::map<WordVector, size_t> overlap;
    for (auto it = mergedRefNGramCounts.cbegin();
         it != mergedRefNGramCounts.cend();
         ++it)
    {
        if (translationNGramCounts[it->first])
        {
          overlap[it->first] = std::min(translationNGramCounts[it->first],
                                        it->second);
        }
    }

    for (auto it = overlap.cbegin(); it != overlap.cend(); ++it)
    {
      matchesByOrder[it->first.size() - 1] += it->second;
    }

    for (size_t order = 1; order < maxOrder + 1; ++order)
    {
      size_t possibleMatches = trIt->size() - order + 1;
      if (possibleMatches > 0)
      {
        possibleMatchesByOrder[order - 1] += possibleMatches;
      }
    }
  }

  precisions = PrecisionType(maxOrder, 0.0);
  ElemType minPrecision = std::numeric_limits<ElemType>::max();
  for (size_t i = 0; i < maxOrder; ++i)
  {
    if (smooth)
      precisions[i]
          = (matchesByOrder[i] + 1.0) / (possibleMatchesByOrder[i] + 1.0);
    else
    {
      if (possibleMatchesByOrder[i] > 0.0)
      {
        precisions[i] = ElemType(matchesByOrder[i]) / possibleMatchesByOrder[i];
      }
      else
        precisions[i] = 0.0;
    }
    if (minPrecision > precisions[i])
      minPrecision = precisions[i];
  }

  ElemType geoMean;
  if (minPrecision > 0)
  {
    ElemType pLogSum = 0.0;
    for (size_t i = 0; i < precisions.size(); ++i)
    {
      pLogSum += (1.0 / maxOrder) * std::log(precisions[i]);
    }
    geoMean = std::exp(pLogSum);
  }
  else
    geoMean = 0.0;
  ratio = ElemType(translationLength) / referenceLength;
  brevityPenalty = (ratio > 1.0) ? 1.0 : std::exp(1.0 - 1.0 / ratio);
  bleuScore = geoMean * brevityPenalty;
  return bleuScore;
}

} // namespace metric
} // namespace mlpack

#endif
