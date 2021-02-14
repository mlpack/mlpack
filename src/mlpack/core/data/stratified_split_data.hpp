/**
 * @file core/data/split_data.hpp
 * @author Anush Kini
 *
 * Defines StratifiedSplit(), a utility function to stratify a dataset into a
 * training set and a test set so that the ratio of each class in the training
 * and test sets is the same as in the original dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_STRATIFIED_SPLIT_DATA_HPP
#define MLPACK_CORE_DATA_STRATIFIED_SPLIT_DATA_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

/**
 * Given an input dataset and labels, stratify into a training set and test set.
 * It is recommended to have the input labels between the range [0, n) where n
 * is the number of different labels. The NormalizeLabels() function in
 * mlpack::data can be used for this.
 * Example usage below. This overload places the stratified dataset into the
 * four output parameters given (trainData, testData, trainLabel,
 * and testLabel).
 *
 * @code
 * arma::mat input = loadData();
 * arma::Row<size_t> label = loadLabel();
 * arma::mat trainData;
 * arma::mat testData;
 * arma::Row<size_t> trainLabel;
 * arma::Row<size_t> testLabel;
 * math::RandomSeed(100); // Set the seed if you like.
 *
 * // Stratify the dataset into a training and test set, with 30% of the data
 * // being held out for the test set.
 * StratifiedSplit(input, label, trainData,
 *                 testData, trainLabel, testLabel, 0.3);
 * @endcode
 *
 * @param input Input dataset to stratify.
 * @param inputLabel Input labels to stratify.
 * @param trainData Matrix to store training data into.
 * @param testData Matrix to store test data into.
 * @param trainLabel Vector to store training labels into.
 * @param testLabel Vector to store test labels into.
 * @param testRatio Percentage of dataset to use for test set (between 0 and 1).
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *     sample is visited in linear order. (Default true.)
 */
template<typename T, typename U>
void StratifiedSplit(const arma::Mat<T>& input,
                     const arma::Row<U>& inputLabel,
                     arma::Mat<T>& trainData,
                     arma::Mat<T>& testData,
                     arma::Row<U>& trainLabel,
                     arma::Row<U>& testLabel,
                     const double testRatio,
                     const bool shuffleData = true)
{
  size_t trainIdx = 0;
  size_t testIdx = 0;
  size_t trainSize = 0;
  size_t testSize = 0;
  arma::uvec labelCounts;
  arma::uvec testLabelCounts;
  U maxLabel = inputLabel.max();

  labelCounts.zeros(maxLabel+1);
  testLabelCounts.zeros(maxLabel+1);

  arma::uvec order =
      arma::linspace<arma::uvec>(0, input.n_cols - 1, input.n_cols);

  if (shuffleData)
  {
    order = arma::shuffle(order);
  }

  for (U label : inputLabel)
  {
    ++labelCounts[label];
  }

  for (arma::uword labelCount : labelCounts)
  {
    testSize += floor(labelCount * testRatio);
    trainSize += labelCount - floor(labelCount * testRatio);
  }

  trainData.set_size(input.n_rows, trainSize);
  testData.set_size(input.n_rows, testSize);
  trainLabel.set_size(trainSize);
  testLabel.set_size(testSize);

  for (arma::uword i : order)
  {
    U label = inputLabel[i];
    if (testLabelCounts[label] < floor(labelCounts[label] * testRatio))
    {
      testLabelCounts[label] += 1;
      testData.col(testIdx) = input.col(i);
      testLabel[testIdx] = inputLabel[i];
      testIdx += 1;
    }
    else
    {
      trainData.col(trainIdx) = input.col(i);
      trainLabel[trainIdx] = inputLabel[i];
      trainIdx += 1;
    }
  }
}

/**
 * Given an input dataset and labels, stratify into a training set and test set.
 * Example usage below.  This overload returns the stratified
 * dataset as a std::tuple with four elements: an arma::Mat<T> containing the
 * training data, an arma::Mat<T> containing the test data, an arma::Row<U>
 * containing the training labels, and an arma::Row<U> containing the test
 * labels.
 *
 * @code
 * arma::mat input = loadData();
 * arma::Row<size_t> label = loadLabel();
 * auto splitResult = StratifiedSplit(input, label, 0.2);
 * @endcode
 *
 * @param input Input dataset to split.
 * @param inputLabel Input labels to split.
 * @param testRatio Percentage of dataset to use for test set (between 0 and 1).
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *     sample is visited in linear order. (Default true).
 * @param stratifyData If true, the train and test splits are stratified
 *
 * @return std::tuple containing trainData (arma::Mat<T>), testData
 *      (arma::Mat<T>), trainLabel (arma::Row<U>), and testLabel (arma::Row<U>).
 */
template<typename T, typename U>
std::tuple<arma::Mat<T>, arma::Mat<T>, arma::Row<U>, arma::Row<U>>
StratifiedSplit(const arma::Mat<T>& input,
                const arma::Row<U>& inputLabel,
                const double testRatio,
                const bool shuffleData = true)
{
  arma::Mat<T> trainData;
  arma::Mat<T> testData;
  arma::Row<U> trainLabel;
  arma::Row<U> testLabel;

  StratifiedSplit(input, inputLabel, trainData, testData, trainLabel, testLabel,
      testRatio, shuffleData);

  return std::make_tuple(std::move(trainData),
                         std::move(testData),
                         std::move(trainLabel),
                         std::move(testLabel));
}

} // namespace data
} // namespace mlpack

#endif
