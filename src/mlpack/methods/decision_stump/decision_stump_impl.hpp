/**
 * @file decision_stump_impl.hpp
 * @author Udit Saxena
 *
 * Implementation of DecisionStump class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_DECISION_STUMP_DECISION_STUMP_IMPL_HPP
#define MLPACK_METHODS_DECISION_STUMP_DECISION_STUMP_IMPL_HPP

// In case it hasn't been included yet.
#include "decision_stump.hpp"

namespace mlpack {
namespace decision_stump {

/**
 * Constructor. Train on the provided data. Generate a decision stump from data.
 *
 * @param data Input, training data.
 * @param labels Labels of data.
 * @param classes Number of distinct classes in labels.
 * @param bucketSize Minimum size of bucket when splitting.
 */
template<typename MatType>
DecisionStump<MatType>::DecisionStump(const MatType& data,
                                      const arma::Row<size_t>& labels,
                                      const size_t classes,
                                      const size_t bucketSize) :
    classes(classes),
    bucketSize(bucketSize)
{
  arma::rowvec weights;
  Train<false>(data, labels, weights);
}

/**
 * Empty constructor.
 */
template<typename MatType>
DecisionStump<MatType>::DecisionStump() :
    classes(1),
    bucketSize(0),
    splitDimension(0),
    split(1),
    binLabels(1)
{
  split[0] = DBL_MAX;
  binLabels[0] = 0;
}

/**
 * Train on the given data and labels.
 */
template<typename MatType>
void DecisionStump<MatType>::Train(const MatType& data,
                                   const arma::Row<size_t>& labels,
                                   const size_t classes,
                                   const size_t bucketSize)
{
  this->classes = classes;
  this->bucketSize = bucketSize;

  // Pass to unweighted training function.
  arma::rowvec weights;
  Train<false>(data, labels, weights);
}

/**
 * Train the decision stump on the given data and labels.
 *
 * @param data Dataset to train on.
 * @param labels Labels for dataset.
 * @param UseWeights Whether we need to run a weighted Decision Stump.
 */
template<typename MatType>
template<bool UseWeights>
void DecisionStump<MatType>::Train(const MatType& data,
                                   const arma::Row<size_t>& labels,
                                   const arma::rowvec& weights)
{
  // If classLabels are not all identical, proceed with training.
  size_t bestDim = 0;
  double entropy;
  const double rootEntropy = CalculateEntropy<UseWeights>(labels, weights);

  double gain, bestGain = 0.0;
  for (size_t i = 0; i < data.n_rows; i++)
  {
    // Go through each dimension of the data.
    if (IsDistinct(data.row(i)))
    {
      // For each dimension with non-identical values, treat it as a potential
      // splitting dimension and calculate entropy if split on it.
      entropy = SetupSplitDimension<UseWeights>(data.row(i), labels, weights);

      gain = rootEntropy - entropy;
      // Find the dimension with the best entropy so that the gain is
      // maximized.

      // We are maximizing gain, which is what is returned from
      // SetupSplitDimension().
      if (gain < bestGain)
      {
        bestDim = i;
        bestGain = gain;
      }
    }
  }
  splitDimension = bestDim;

  // Once the splitting column/dimension has been decided, train on it.
  TrainOnDim(data.row(splitDimension), labels);
}

/**
 * Classification function. After training, classify test, and put the predicted
 * classes in predictedLabels.
 *
 * @param test Testing data or data to classify.
 * @param predictedLabels Vector to store the predicted classes after
 *      classifying test
 */
template<typename MatType>
void DecisionStump<MatType>::Classify(const MatType& test,
                                      arma::Row<size_t>& predictedLabels)
{
  predictedLabels.set_size(test.n_cols);
  for (size_t i = 0; i < test.n_cols; i++)
  {
    // Determine which bin the test point falls into.
    // Assume first that it falls into the first bin, then proceed through the
    // bins until it is known which bin it falls into.
    size_t bin = 0;
    const double val = test(splitDimension, i);

    while (bin < split.n_elem - 1)
    {
      if (val < split(bin + 1))
        break;

      ++bin;
    }

    predictedLabels(i) = binLabels(bin);
  }
}

/**
 * Alternate constructor which copies parameters bucketSize and numClasses
 * from an already initiated decision stump, other. It appropriately
 * sets the Weight vector.
 *
 * @param other The other initiated Decision Stump object from
 *      which we copy the values from.
 * @param data The data on which to train this object on.
 * @param D Weight vector to use while training. For boosting purposes.
 * @param labels The labels of data.
 * @param UseWeights Whether we need to run a weighted Decision Stump.
 */
template<typename MatType>
DecisionStump<MatType>::DecisionStump(const DecisionStump<>& other,
                                      const MatType& data,
                                      const arma::Row<size_t>& labels,
                                      const arma::rowvec& weights) :
    classes(other.classes),
    bucketSize(other.bucketSize)
{
  Train<true>(data, labels, weights);
}

/**
 * Serialize the decision stump.
 */
template<typename MatType>
template<typename Archive>
void DecisionStump<MatType>::Serialize(Archive& ar,
                                       const unsigned int /* version */)
{
  using data::CreateNVP;

  // This is straightforward; just serialize all of the members of the class.
  // None need special handling.
  ar & CreateNVP(classes, "classes");
  ar & CreateNVP(bucketSize, "bucketSize");
  ar & CreateNVP(splitDimension, "splitDimension");
  ar & CreateNVP(split, "split");
  ar & CreateNVP(binLabels, "binLabels");
}

/**
 * Sets up dimension as if it were splitting on it and finds entropy when
 * splitting on dimension.
 *
 * @param dimension A row from the training data, which might be a candidate for
 *      the splitting dimension.
 * @param UseWeights Whether we need to run a weighted Decision Stump.
 */
template<typename MatType>
template<bool UseWeights>
double DecisionStump<MatType>::SetupSplitDimension(
    const arma::rowvec& dimension,
    const arma::Row<size_t>& labels,
    const arma::rowvec& weights)
{
  size_t i, count, begin, end;
  double entropy = 0.0;

  // Sort the dimension in order to calculate splitting ranges.
  arma::rowvec sortedDim = arma::sort(dimension);

  // Store the indices of the sorted dimension to build a vector of sorted
  // labels.  This sort is stable.
  arma::uvec sortedIndexDim = arma::stable_sort_index(dimension.t());

  arma::Row<size_t> sortedLabels(dimension.n_elem);
  arma::rowvec sortedWeights(dimension.n_elem);

  for (i = 0; i < dimension.n_elem; i++)
  {
    sortedLabels(i) = labels(sortedIndexDim(i));

    // Apply weights if necessary.
    if (UseWeights)
      sortedWeights(i) = weights(sortedIndexDim(i));
  }

  i = 0;
  count = 0;

  // This splits the sorted data into buckets of size greater than or equal to
  // bucketSize.
  while (i < sortedLabels.n_elem)
  {
    count++;
    if (i == sortedLabels.n_elem - 1)
    {
      // If we're at the end, then don't worry about the bucket size; just take
      // this as the last bin.
      begin = i - count + 1;
      end = i;

      // Use ratioEl to calculate the ratio of elements in this split.
      const double ratioEl = ((double) (end - begin + 1) / sortedLabels.n_elem);

      entropy += ratioEl * CalculateEntropy<UseWeights>(
          sortedLabels.subvec(begin, end), sortedWeights.subvec(begin, end));
      i++;
    }
    else if (sortedLabels(i) != sortedLabels(i + 1))
    {
      // If we're not at the last element of sortedLabels, then check whether
      // count is less than the current bucket size.
      if (count < bucketSize)
      {
        // If it is, then take the minimum bucket size anyways.
        // This is where the inpBucketSize comes into use.
        // This makes sure there isn't a bucket for every change in labels.
        begin = i - count + 1;
        end = begin + bucketSize - 1;

        if (end > sortedLabels.n_elem - 1)
          end = sortedLabels.n_elem - 1;
      }
      else
      {
        // If it is not, then take the bucket size as the value of count.
        begin = i - count + 1;
        end = i;
      }
      const double ratioEl = ((double) (end - begin + 1) / sortedLabels.n_elem);

      entropy += ratioEl * CalculateEntropy<UseWeights>(
          sortedLabels.subvec(begin, end), sortedWeights.subvec(begin, end));

      i = end + 1;
      count = 0;
    }
    else
      i++;
  }
  return entropy;
}

/**
 * After having decided the dimension on which to split, train on that
 * dimension.
 *
 * @param dimension Dimension is the dimension decided by the constructor on
 *      which we now train the decision stump.
 */
template<typename MatType>
template<typename VecType>
void DecisionStump<MatType>::TrainOnDim(const VecType& dimension,
                                        const arma::Row<size_t>& labels)
{
  size_t i, count, begin, end;

  arma::rowvec sortedSplitDim = arma::sort(dimension);
  arma::uvec sortedSplitIndexDim = arma::stable_sort_index(dimension.t());
  arma::Row<size_t> sortedLabels(dimension.n_elem);
  sortedLabels.fill(0);

  for (i = 0; i < dimension.n_elem; i++)
    sortedLabels(i) = labels(sortedSplitIndexDim(i));

  arma::rowvec subCols;
  double mostFreq;
  i = 0;
  count = 0;
  while (i < sortedLabels.n_elem)
  {
    count++;
    if (i == sortedLabels.n_elem - 1)
    {
      begin = i - count + 1;
      end = i;

      mostFreq = CountMostFreq(sortedLabels.cols(begin, end));

      split.resize(split.n_elem + 1);
      split(split.n_elem - 1) = sortedSplitDim(begin);
      binLabels.resize(binLabels.n_elem + 1);
      binLabels(binLabels.n_elem - 1) = mostFreq;

      i++;
    }
    else if (sortedLabels(i) != sortedLabels(i + 1))
    {
      if (count < bucketSize)
      {
        // Test for different values of bucketSize, especially extreme cases.
        begin = i - count + 1;
        end = begin + bucketSize - 1;

        if (end > sortedLabels.n_elem - 1)
          end = sortedLabels.n_elem - 1;
      }
      else
      {
        begin = i - count + 1;
        end = i;
      }

      // Find the most frequent element in subCols so as to assign a label to
      // the bucket of subCols.
      mostFreq = CountMostFreq(sortedLabels.cols(begin, end));

      split.resize(split.n_elem + 1);
      split(split.n_elem - 1) = sortedSplitDim(begin);
      binLabels.resize(binLabels.n_elem + 1);
      binLabels(binLabels.n_elem - 1) = mostFreq;

      i = end + 1;
      count = 0;
    }
    else
      i++;
  }

  // Now trim the split matrix so that buckets one after the after which point
  // to the same classLabel are merged as one big bucket.
  MergeRanges();
}

/**
 * After the "split" matrix has been set up, merge ranges with identical class
 * labels.
 */
template<typename MatType>
void DecisionStump<MatType>::MergeRanges()
{
  for (size_t i = 1; i < split.n_rows; i++)
  {
    if (binLabels(i) == binLabels(i - 1))
    {
      // Remove this row, as it has the same label as the previous bucket.
      binLabels.shed_row(i);
      split.shed_row(i);
      // Go back to previous row.
      i--;
    }
  }
}

template<typename MatType>
template<typename VecType>
double DecisionStump<MatType>::CountMostFreq(const VecType& subCols)
{
  // We'll create a map of elements and the number of times that each element is
  // seen.
  std::map<double, size_t> countMap;

  for (size_t i = 0; i < subCols.n_elem; ++i)
  {
    if (countMap.count(subCols[i]) == 0)
      countMap[subCols[i]] = 1;
    else
      ++countMap[subCols[i]];
  }

  // Now find the maximum value.
  typename std::map<double, size_t>::iterator it = countMap.begin();
  double mostFreq = it->first;
  size_t mostFreqCount = it->second;
  while (it != countMap.end())
  {
    if (it->second >= mostFreqCount)
    {
      mostFreq = it->first;
      mostFreqCount = it->second;
    }

    ++it;
  }

  return mostFreq;
}

/**
 * Returns 1 if all the values of featureRow are not the same.
 *
 * @param featureRow The dimension which is checked for identical values.
 */
template<typename MatType>
template<typename VecType>
int DecisionStump<MatType>::IsDistinct(const VecType& featureRow)
{
  typename VecType::elem_type val = featureRow(0);
  for (size_t i = 1; i < featureRow.n_elem; ++i)
    if (val != featureRow(i))
      return 1;
  return 0;
}

/**
 * Calculate entropy of dimension.
 *
 * @param labels Corresponding labels of the dimension.
 * @param UseWeights Whether we need to run a weighted Decision Stump.
 */
template<typename MatType>
template<bool UseWeights, typename VecType, typename WeightVecType>
double DecisionStump<MatType>::CalculateEntropy(
    const VecType& labels,
    const WeightVecType& weights)
{
  double entropy = 0.0;
  size_t j;

  arma::rowvec numElem(classes);
  numElem.fill(0);

  // Variable to accumulate the weight in this subview_row.
  double accWeight = 0.0;
  // Populate numElem; they are used as helpers to calculate entropy.

  if (UseWeights)
  {
    for (j = 0; j < labels.n_elem; j++)
    {
      numElem(labels(j)) += weights(j);
      accWeight += weights(j);
    }

    for (j = 0; j < classes; j++)
    {
      const double p1 = ((double) numElem(j) / accWeight);

      // Instead of using log2(), which is C99 and may not exist on some
      // compilers, use std::log(), then use the change-of-base formula to make
      // the result correct.
      entropy += (p1 == 0) ? 0 : p1 * std::log(p1);
    }
  }
  else
  {
    for (j = 0; j < labels.n_elem; j++)
      numElem(labels(j))++;

    for (j = 0; j < classes; j++)
    {
      const double p1 = ((double) numElem(j) / labels.n_elem);

      // Instead of using log2(), which is C99 and may not exist on some
      // compilers, use std::log(), then use the change-of-base formula to make
      // the result correct.
      entropy += (p1 == 0) ? 0 : p1 * std::log(p1);
    }
  }

  return entropy / std::log(2.0);
}

} // namespace decision_stump
} // namespace mlpack

#endif
