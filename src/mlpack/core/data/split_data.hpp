/**
 * @file core/data/split_data.hpp
 * @author Tham Ngap Wei, Keon Kim
 *
 * Defines Split(), a utility function to split a dataset into a
 * training set and a test set.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SPLIT_DATA_HPP
#define MLPACK_CORE_DATA_SPLIT_DATA_HPP

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
  /**
   * Basic idea:
   * Let us say we have to stratify a dataset based on labels:
   * 0 0 0 0 0 (5 0s)
   * 1 1 1 1 1 1 1 1 1 1 1 (11 1s)
   *
   * Let our test ratio be 0.2.
   * Then, the number of 0 labels in our test set = floor(5 * 0.2) = 1.
   * The number of 1 labels in our test set = floor(11 * 0.2) = 2.
   *
   * In our first pass over the dataset,
   * We visit each label and keep count of each label in our 'labelCounts' uvec.
   *
   * We then take a second pass over the dataset.
   * We now maintain an additional uvec 'testLabelCounts' to hold the label
   * counts of our test set.
   *
   * In this pass, when we encounter a label we check the 'testLabelCounts' uvec
   * for the count of this label in the test set.
   * If this count is less than the required number of labels in the test set,
   * we add the data to the test set and increment the label count in the uvec.
   * If this count is equal to or more than the required count in the test set,
   * we add this data to the train set.
   *
   * Based on the above steps, we get the following labels in the split set:
   * Train set (4 0s, 9 1s)
   * 0 0 0 0
   * 1 1 1 1 1 1 1 1 1
   *
   * Test set (1 0s, 2 1s)
   * 0
   * 1 1
   */
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
 * Given an input dataset and labels, split into a training set and test set.
 * Example usage below.  This overload places the split dataset into the four
 * output parameters given (trainData, testData, trainLabel, and testLabel).
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
 * // Split the dataset into a training and test set, with 30% of the data being
 * // held out for the test set.
 * Split(input, label, trainData,
 *                testData, trainLabel, testLabel, 0.3);
 * @endcode
 *
 * @tparam LabelsType Type of input labels. It must be arma::Mat or arma::row.
 * @param input Input dataset to split.
 * @param inputLabel Input labels to split.
 * @param trainData Matrix to store training data into.
 * @param testData Matrix to store test data into.
 * @param trainLabel Vector to store training labels into.
 * @param testLabel Vector to store test labels into.
 * @param testRatio Percentage of dataset to use for test set (between 0 and 1).
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *       sample is visited in linear order. (Default true.)
 */
template<typename T, typename LabelsType,
         typename = std::enable_if_t<arma::is_Row<LabelsType>::value ||
                          arma::is_Mat_only<LabelsType>::value> >
void Split(const arma::Mat<T>& input,
           const LabelsType& inputLabel,
           arma::Mat<T>& trainData,
           arma::Mat<T>& testData,
           LabelsType& trainLabel,
           LabelsType& testLabel,
           const double testRatio,
           const bool shuffleData = true)
{
  const size_t testSize = static_cast<size_t>(input.n_cols * testRatio);
  const size_t trainSize = input.n_cols - testSize;
  trainData.set_size(input.n_rows, trainSize);
  testData.set_size(input.n_rows, testSize);
  trainLabel.set_size(trainSize);
  testLabel.set_size(testSize);

  if (shuffleData)
  {
    arma::uvec order = arma::shuffle(arma::linspace<arma::uvec>(
        0, input.n_cols - 1, input.n_cols));
    if (trainSize > 0)
    {
      trainData = input.cols(order.subvec(0, trainSize - 1));
      trainLabel = inputLabel.cols(order.subvec(0, trainSize - 1));
    }
    if (trainSize < input.n_cols)
    {
    testData = input.cols(order.subvec(trainSize, input.n_cols - 1));
    testLabel = inputLabel.cols(order.subvec(trainSize, input.n_cols - 1));
    }
  }
  else
  {
    if (trainSize > 0)
    {
      trainData = input.cols(0, trainSize - 1);
      trainLabel = inputLabel.subvec(0, trainSize - 1);
    }
    if (trainSize < input.n_cols)
    {
      testData = input.cols(trainSize , input.n_cols - 1);
      testLabel = inputLabel.subvec(trainSize , input.n_cols - 1);
    }
  }
}

/**
 * Given an input dataset, split into a training set and test set.
 * Example usage below. This overload places the split dataset into the two
 * output parameters given (trainData, testData).
 *
 * @code
 * arma::mat input = loadData();
 * arma::mat trainData;
 * arma::mat testData;
 * math::RandomSeed(100); // Set the seed if you like.
 *
 * // Split the dataset into a training and test set, with 30% of the data being
 * // held out for the test set.
 * Split(input, trainData, testData, 0.3);
 * @endcode
 *
 * @param input Input dataset to split.
 * @param trainData Matrix to store training data into.
 * @param testData Matrix to store test data into.
 * @param testRatio Percentage of dataset to use for test set (between 0 and 1).
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *       sample is visited in linear order. (Default true).
 */
template<typename T>
void Split(const arma::Mat<T>& input,
           arma::Mat<T>& trainData,
           arma::Mat<T>& testData,
           const double testRatio,
           const bool shuffleData = true)
{
  const size_t testSize = static_cast<size_t>(input.n_cols * testRatio);
  const size_t trainSize = input.n_cols - testSize;
  trainData.set_size(input.n_rows, trainSize);
  testData.set_size(input.n_rows, testSize);

  if (shuffleData)
  {
    arma::uvec order = arma::shuffle(arma::linspace<arma::uvec>(
        0, input.n_cols - 1, input.n_cols));

    if (trainSize > 0)
      trainData = input.cols(order.subvec(0, trainSize - 1));

    if (trainSize < input.n_cols)
      testData = input.cols(order.subvec(trainSize, input.n_cols - 1));
  }
  else
  {
    if (trainSize > 0)
      trainData = input.cols(0, trainSize - 1);

    if (trainSize < input.n_cols)
      testData = input.cols(trainSize , input.n_cols - 1);
  }
}

/**
 * Given an input dataset and labels, split into a training set and test set.
 * Example usage below.  This overload returns the split dataset as a std::tuple
 * with four elements: an arma::Mat<T> containing the training data, an
 * arma::Mat<T> containing the test data, an arma::Row<U> containing the
 * training labels, and an arma::Row<U> containing the test labels.
 *
 * @code
 * arma::mat input = loadData();
 * arma::Row<size_t> label = loadLabel();
 * auto splitResult = Split(input, label, 0.2);
 * @endcode
 *
 * @tparam LabelsType Type of input labels. It must be arma::Mat or arma::row.
 * @param input Input dataset to split.
 * @param inputLabel Input labels to split.
 * @param testRatio Percentage of dataset to use for test set (between 0 and 1).
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *     sample is visited in linear order. (Default true).
 * @param stratifyData If true, the train and test splits are stratified
 *     so that the ratio of each class in the training and test sets is the same
 *     as in the original dataset.
 * @return std::tuple containing trainData (arma::Mat<T>), testData
 *      (arma::Mat<T>), trainLabel (arma::Row<U>), and testLabel (arma::Row<U>).
 */
template<typename T, typename LabelsType,
         typename = std::enable_if_t<arma::is_Row<LabelsType>::value ||
                          arma::is_Mat_only<LabelsType>::value> >
std::tuple<arma::Mat<T>, arma::Mat<T>, LabelsType, LabelsType>
Split(const arma::Mat<T>& input,
      const LabelsType& inputLabel,
      const double testRatio,
      const bool shuffleData = true,
      const bool stratifyData = false)
{
  arma::Mat<T> trainData;
  arma::Mat<T> testData;
  LabelsType trainLabel;
  LabelsType testLabel;

  if (stratifyData)
  {
    StratifiedSplit(input, inputLabel, trainData, testData, trainLabel,
        testLabel, testRatio, shuffleData);
  }
  else
  {
    Split(input, inputLabel, trainData, testData, trainLabel, testLabel,
        testRatio, shuffleData);
  }

  return std::make_tuple(std::move(trainData),
                         std::move(testData),
                         std::move(trainLabel),
                         std::move(testLabel));
}

/**
 * Given an input dataset, split into a training set and test set.
 * Example usage below.  This overload returns the split dataset as a std::tuple
 * with two elements: an arma::Mat<T> containing the training data and an
 * arma::Mat<T> containing the test data.
 *
 * @code
 * arma::mat input = loadData();
 * auto splitResult = Split(input, 0.2);
 * @endcode
 *
 * @param input Input dataset to split.
 * @param testRatio Percentage of dataset to use for test set (between 0 and 1).
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *       sample is visited in linear order. (Default true).
 * @return std::tuple containing trainData (arma::Mat<T>)
 *      and testData (arma::Mat<T>).
 */
template<typename T>
std::tuple<arma::Mat<T>, arma::Mat<T>>
Split(const arma::Mat<T>& input,
      const double testRatio,
      const bool shuffleData = true)
{
  arma::Mat<T> trainData;
  arma::Mat<T> testData;
  Split(input, trainData, testData, testRatio, shuffleData);

  return std::make_tuple(std::move(trainData),
                         std::move(testData));
}

} // namespace data
} // namespace mlpack

#endif
