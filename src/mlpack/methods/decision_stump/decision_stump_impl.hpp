/**
 * @file decision_stump_impl.hpp
 * @author Udit Saxena
 *
 * Implementation of DecisionStump class.
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
  arma::Row<size_t> zLabels(labels.n_elem);
  zLabels.fill(0);
  classLabels = labels + zLabels;

  numClass = classes;
  bucketSize = inpBucketSize;

  /* Check whether the input labels are not all identical. */
  if (!isDistinct<size_t>(classLabels))
  {
    // If the classLabels are all identical, the default class is the only
    // class.
    oneClass = 1;
    defaultClass = classLabels(0);
  }

  else
  {
    // If classLabels are not all identical, proceed with training.
    oneClass = 0;
    int bestAtt = -1;
    double entropy;
    double bestEntropy = DBL_MAX;

    // Set the default class to handle attribute values which are not present in
    // the training data.
    defaultClass = CountMostFreq<size_t>(classLabels);

    for (int i = 0; i < data.n_rows; i++)
    {
      // Go through each attribute of the data.
      if (isDistinct<double>(data.row(i)))
      {
        // For each attribute with non-identical values, treat it as a potential
        // splitting attribute and calculate entropy if split on it.
        entropy = SetupSplitAttribute(data.row(i));

        // Find the attribute with the bestEntropy so that the gain is
        // maximized.
        if (entropy < bestEntropy)
        {
          bestAtt = i;
          bestEntropy = entropy;
        }
      }
    }
    splitCol = bestAtt;

    // Once the splitting column/attribute has been decided, train on it.
    TrainOnAtt<double>(data.row(splitCol));
  }
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
  int flag;
  double val;
  if (!oneClass)
  {
    for (int i = 0; i < test.n_cols; i++)
    {
      int j = 0;
      flag = 0;

      val = test(splitCol,i);
      while ((j < split.n_rows) && (!flag))
      {
        if (val < split(j, 0) && (!j))
        {
          predictedLabels(i) = split(0, 1);
          flag = 1;
        }
        else if (val >= split(j, 0))
        {
          if (j == split.n_rows - 1)
          {
            predictedLabels(i) = split(split.n_rows - 1, 1);
            flag = 1;
          }
          else if (val < split(j + 1, 0))
          {
            predictedLabels(i) = split(j, 1);
            flag = 1;
          }
        }
        j++;
      }
    }
  }
  else
  {
    for (int i = 0; i < test.n_cols; i++)
      predictedLabels(i) = defaultClass;
  }
}

/**
 * Sets up attribute as if it were splitting on it and finds entropy when
 * splitting on attribute.
 *
 * @param attribute A row from the training data, which might be a candidate for
 *      the splitting attribute.
 */
template <typename MatType>
double DecisionStump<MatType>::SetupSplitAttribute(const arma::rowvec& attribute)
{
  int i, count, begin, end;
  double entropy = 0.0;

  // Sort the attribute in order to calculate splitting ranges.
  arma::rowvec sortedAtt = arma::sort(attribute);

  // Store the indices of the sorted attribute to build a vector of sorted
  // labels.  This sort is stable.
  arma::uvec sortedIndexAtt = arma::stable_sort_index(attribute.t());

  arma::Row<size_t> sortedLabels(attribute.n_elem);
  sortedLabels.fill(0);

  for (i = 0; i < attribute.n_elem; i++)
    sortedLabels(i) = classLabels(sortedIndexAtt(i));

  arma::rowvec subColLabels;
  arma::rowvec subColAtts;

  i = 0;
  count = 0;

  // This splits the sorted into buckets of size greater than or equal to
  // inpBucketSize.
  while (i < sortedLabels.n_elem)
  {
    count++;
    if (i == sortedLabels.n_elem - 1)
    {
      begin = i - count + 1;
      end = i;

      arma::rowvec zSubColLabels((sortedLabels.cols(begin, end)).n_elem);
      zSubColLabels.fill(0.0);

      arma::rowvec zSubColAtts((sortedAtt.cols(begin, end)).n_elem);
      zSubColAtts.fill(0.0);

      subColLabels = sortedLabels.cols(begin, end) + zSubColLabels;

      subColAtts = sortedAtt.cols(begin, end) + zSubColAtts;

      entropy += CalculateEntropy(subColAtts, subColLabels);
      i++;
    }
    else if (sortedLabels(i) != sortedLabels(i + 1))
    {
      if (count < bucketSize)
      {
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

      arma::rowvec zSubColLabels((sortedLabels.cols(begin, end)).n_elem);
      zSubColLabels.fill(0.0);

      arma::rowvec zSubColAtts((sortedAtt.cols(begin, end)).n_elem);
      zSubColAtts.fill(0.0);

      subColLabels = sortedLabels.cols(begin, end) + zSubColLabels;

      subColAtts = sortedAtt.cols(begin, end) + zSubColAtts;

      // Now use subColLabels and subColAtts to calculate entropy.
      entropy += CalculateEntropy(subColAtts, subColLabels);

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
void DecisionStump<MatType>::TrainOnAtt(const arma::rowvec& attribute)
{
  int i, count, begin, end;

  arma::rowvec sortedSplitAtt = arma::sort(attribute);
  arma::uvec sortedSplitIndexAtt = arma::stable_sort_index(attribute.t());
  arma::Row<size_t> sortedLabels(attribute.n_elem);
  sortedLabels.fill(0);
  arma::mat tempSplit;

  for (i = 0; i < attribute.n_elem; i++)
    sortedLabels(i) = classLabels(sortedSplitIndexAtt(i));

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

      tempSplit << sortedSplitAtt(begin)<< mostFreq << arma::endr;
      split = arma::join_cols(split, tempSplit);

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

      tempSplit << sortedSplitAtt(begin) << mostFreq << arma::endr;
      split = arma::join_cols(split, tempSplit);

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
  for (int i = 1; i < split.n_rows; i++)
  {
    if (split(i, 1) == split(i - 1, 1))
    {
      // Remove this row, as it has the same label as the previous bucket.
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
  rType element;
  int count = 0, localCount = 0;

  // An O(n) loop which counts the most frequent element in sortCounts
  for (int i = 0; i < sortCounts.n_elem; ++i)
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
int DecisionStump<MatType>::isDistinct(const arma::Row<rType>& featureRow)
{
  if (featureRow.max() - featureRow.min() > 0)
    return 1;
  else
    return 0;
}

/**
 * Calculating Entropy of attribute.
 *
 * @param attribute The attribute for which we calculate the entropy.
 * @param labels Corresponding labels of the attribute.
 */
template<typename MatType>
double DecisionStump<MatType>::CalculateEntropy(const arma::rowvec& attribute,
                                                const arma::rowvec& labels)
{
  double entropy = 0.0;

  arma::rowvec uniqueAtt = arma::unique(attribute);
  arma::rowvec uniqueLabel = arma::unique(labels);
  arma::Row<size_t> numElem(uniqueAtt.n_elem);
  numElem.fill(0);
  arma::Mat<size_t> entropyArray(uniqueAtt.n_elem,numClass);
  entropyArray.fill(0);

  // Populate entropyArray and numElem; they are used as helpers to calculate
  // entropy.
  for (int j = 0; j < uniqueAtt.n_elem; j++)
  {
    for (int i = 0; i < attribute.n_elem; i++)
    {
      if (uniqueAtt[j] == attribute[i])
      {
        entropyArray(j,labels(i))++;
        numElem(j)++;
      }
    }
  }

  for (int j = 0; j < uniqueAtt.size(); j++)
  {
    const double p1 = ((double) numElem(j) / attribute.n_elem);

    for (int i = 0; i < numClass; i++)
    {
      const double p2 = ((double) entropyArray(j, i) / numElem(j));
      const double p3 = (p2 == 0) ? 0 : p2 * log2(p2);

      entropy += p1 * p3;
    }
  }

  return entropy;
}

}; // namespace decision_stump
}; // namespace mlpack

#endif
