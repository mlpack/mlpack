/**
 * @file decision_stump_impl.hpp
 * @author Udit Saxena
 *
 * Implementation of DecisionStump class.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __MLPACK_METHODS_DECISION_STUMP_DECISION_STUMP_IMPL_HPP
#define __MLPACK_METHODS_DECISION_STUMP_DECISION_STUMP_IMPL_HPP

// In case it hasn't been included yet.
#include "decision_stump.hpp"

#include <set>
#include <algorithm>

namespace mlpack {
namespace decision_stump {

/**
 * Constructor. Train on the provided data. Generate a decision stump from data.
 *
 * @param data Input, training data.
 * @param labels Labels of data.
 * @param classes Number of distinct classes in labels.
 * @param inpBucketSize Minimum size of bucket when splitting.
 */
template<typename MatType>
DecisionStump<MatType>::DecisionStump(const MatType& data,
                                      const arma::Row<size_t>& labels,
                                      const size_t classes,
                                      size_t inpBucketSize)
{
  numClass = classes;
  bucketSize = inpBucketSize;

  // If classLabels are not all identical, proceed with training.
  size_t bestAtt = 0;
  double entropy;
  const double rootEntropy = CalculateEntropy<size_t>(
      labels.subvec(0, labels.n_elem - 1));

  double gain, bestGain = 0.0;
  for (size_t i = 0; i < data.n_rows; i++)
  {
    // Go through each attribute of the data.
    if (IsDistinct<double>(data.row(i)))
    {
      // For each attribute with non-identical values, treat it as a potential
      // splitting attribute and calculate entropy if split on it.
      entropy = SetupSplitAttribute(data.row(i), labels);

      // Log::Debug << "Entropy for attribute " << i << " is " << entropy << ".\n";
      gain = rootEntropy - entropy;
      // Find the attribute with the best entropy so that the gain is
      // maximized.

      // if (entropy < bestEntropy)
      // Instead of the above rule, we are maximizing gain, which was
      // what is returned from SetupSplitAttribute.
      if (gain < bestGain)
      {
        bestAtt = i;
        bestGain = gain;
      }
    }
  }
  splitAttribute = bestAtt;

  // Once the splitting column/attribute has been decided, train on it.
  TrainOnAtt<double>(data.row(splitAttribute), labels);
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
  for (size_t i = 0; i < test.n_cols; i++)
  {
    // Determine which bin the test point falls into.
    // Assume first that it falls into the first bin, then proceed through the
    // bins until it is known which bin it falls into.
    size_t bin = 0;
    const double val = test(splitAttribute, i);

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
 *
 *
 *
 *
 *
 */
template <typename MatType>
DecisionStump<MatType>::DecisionStump(const DecisionStump<>& ds)
{
  numClass = ds.numClass;

  splitAttribute = ds.splitAttribute;

  bucketSize = ds.bucketSize;

  split = ds.split;

  binLabels = ds.binLabels;
}

/**
 *
 *
 *
 *
 *
 *
template <typename MatType>
DecisionStump<MatType>::ModifyData(MatType& data, const arma::Row<double>& D)
 */

/**
 * Sets up attribute as if it were splitting on it and finds entropy when
 * splitting on attribute.
 *
 * @param attribute A row from the training data, which might be a candidate for
 *      the splitting attribute.
 */
template <typename MatType>
double DecisionStump<MatType>::SetupSplitAttribute(
    const arma::rowvec& attribute,
    const arma::Row<size_t>& labels)
{
  size_t i, count, begin, end;
  double entropy = 0.0;

  // Sort the attribute in order to calculate splitting ranges.
  arma::rowvec sortedAtt = arma::sort(attribute);

  // Store the indices of the sorted attribute to build a vector of sorted
  // labels.  This sort is stable.
  arma::uvec sortedIndexAtt = arma::stable_sort_index(attribute.t());

  arma::Row<size_t> sortedLabels(attribute.n_elem);
  sortedLabels.fill(0);

  for (i = 0; i < attribute.n_elem; i++)
    sortedLabels(i) = labels(sortedIndexAtt(i));

  i = 0;
  count = 0;

  // This splits the sorted into buckets of size greater than or equal to
  // inpBucketSize.
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

      entropy += ratioEl * CalculateEntropy<size_t>(
          sortedLabels.subvec(begin, end));
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

      entropy += ratioEl * CalculateEntropy<size_t>(
          sortedLabels.subvec(begin, end));

      i = end + 1;
      count = 0;
    }
    else
      i++;
  }
  return entropy;
}

/**
 * After having decided the attribute on which to split, train on that
 * attribute.
 *
 * @param attribute Attribute is the attribute decided by the constructor on
 *      which we now train the decision stump.
 */
template <typename MatType>
template <typename rType>
void DecisionStump<MatType>::TrainOnAtt(const arma::rowvec& attribute,
                                        const arma::Row<size_t>& labels)
{
  size_t i, count, begin, end;

  arma::rowvec sortedSplitAtt = arma::sort(attribute);
  arma::uvec sortedSplitIndexAtt = arma::stable_sort_index(attribute.t());
  arma::Row<size_t> sortedLabels(attribute.n_elem);
  sortedLabels.fill(0);
  arma::vec tempSplit;
  arma::Row<size_t> tempLabel;

  for (i = 0; i < attribute.n_elem; i++)
    sortedLabels(i) = labels(sortedSplitIndexAtt(i));

  arma::rowvec subCols;
  rType mostFreq;
  i = 0;
  count = 0;
  while (i < sortedLabels.n_elem)
  {
    count++;
    if (i == sortedLabels.n_elem - 1)
    {
      begin = i - count + 1;
      end = i;

      arma::rowvec zSubCols((sortedLabels.cols(begin, end)).n_elem);
      zSubCols.fill(0.0);

      subCols = sortedLabels.cols(begin, end) + zSubCols;

      mostFreq = CountMostFreq<double>(subCols);

      split.resize(split.n_elem + 1);
      split(split.n_elem - 1) = sortedSplitAtt(begin);
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
      arma::rowvec zSubCols((sortedLabels.cols(begin, end)).n_elem);
      zSubCols.fill(0.0);

      subCols = sortedLabels.cols(begin, end) + zSubCols;

      // Find the most frequent element in subCols so as to assign a label to
      // the bucket of subCols.
      mostFreq = CountMostFreq<double>(subCols);

      split.resize(split.n_elem + 1);
      split(split.n_elem - 1) = sortedSplitAtt(begin);
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
template <typename MatType>
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

template <typename MatType>
template <typename rType>
rType DecisionStump<MatType>::CountMostFreq(const arma::Row<rType>& subCols)
{
  // Sort subCols for easier processing.
  arma::Row<rType> sortCounts = arma::sort(subCols);
  rType element = sortCounts[0];
  size_t count = 0, localCount = 0;

  if (sortCounts.n_elem == 1)
    return sortCounts[0];

  // An O(n) loop which counts the most frequent element in sortCounts.
  for (size_t i = 0; i < sortCounts.n_elem; ++i)
  {
    if (i == sortCounts.n_elem - 1)
    {
      if (sortCounts(i - 1) == sortCounts(i))
      {
        // element = sortCounts(i - 1);
        localCount++;
      }
      else if (localCount > count)
        count = localCount;
    }
    else if (sortCounts(i) != sortCounts(i + 1))
    {
      localCount = 0;
      count++;
    }
    else
    {
      localCount++;
      if (localCount > count)
      {
        count = localCount;
        if (localCount == 1)
          element = sortCounts(i);
      }
    }
  }
  return element;
}

/**
 * Returns 1 if all the values of featureRow are not same.
 *
 * @param featureRow The attribute which is checked for identical values.
 */
template <typename MatType>
template <typename rType>
int DecisionStump<MatType>::IsDistinct(const arma::Row<rType>& featureRow)
{
  rType val = featureRow(0);
  for (size_t i = 1; i < featureRow.n_elem; ++i)
    if (val != featureRow(i))
      return 1;
  return 0;
}

/**
 * Calculate entropy of attribute.
 *
 * @param attribute The attribute for which we calculate the entropy.
 * @param labels Corresponding labels of the attribute.
 */
template<typename MatType>
template<typename AttType, typename LabelType>
double DecisionStump<MatType>::CalculateEntropy(arma::subview_row<LabelType> labels)
{
  double entropy = 0.0;
  size_t j;

  arma::Row<size_t> numElem(numClass);
  numElem.fill(0);

  // Populate numElem; they are used as helpers to calculate
  // entropy.
  for (j = 0; j < labels.n_elem; j++)
    numElem(labels(j))++;

  // The equation for entropy uses log2(), but log2() is from C99 and thus
  // Visual Studio will not have it.  Therefore, we will use std::log(), and
  // then apply the change-of-base formula at the end of the calculation.
  for (j = 0; j < numClass; j++)
  {
    const double p1 = ((double) numElem(j) / labels.n_elem);

    entropy += (p1 == 0) ? 0 : p1 * std::log(p1);
  }

  return entropy / std::log(2.0);
}

}; // namespace decision_stump
}; // namespace mlpack

#endif
