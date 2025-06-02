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
 * This helper function splits any `input` data into training and testing parts.
 * In order to shuffle the input data before spliting, an array of shuffled
 * indices of the input data is passed in the form of argument `order`.
 */
template<typename InputType>
void SplitHelper(const InputType& input,
                 InputType& train,
                 InputType& test,
                 const double testRatio,
                 const arma::uvec& order = arma::uvec(),
                 const typename std::enable_if_t<
                     !IsCube<InputType>::value && !IsField<InputType>::value
                 >* = 0)
{
  const size_t testSize = static_cast<size_t>(input.n_cols * testRatio);
  const size_t trainSize = input.n_cols - testSize;

  // Initialising the sizes of outputs if not already initialized.
  train.set_size(input.n_rows, trainSize);
  test.set_size(input.n_rows, testSize);

  // Shuffling and splitting simultaneously.
  if (!order.is_empty())
  {
    if (trainSize > 0)
      train = input.cols(order.subvec(0, trainSize - 1));

    if (trainSize < input.n_cols)
      test = input.cols(order.subvec(trainSize, order.n_elem - 1));
  }
  // Splitting only.
  else
  {
    if (trainSize > 0)
      train = input.cols(0, trainSize - 1);

    if (trainSize < input.n_cols)
      test = input.cols(trainSize, input.n_cols - 1);
  }
}

/**
 * This is the same as the helper function above, but for arma::field and
 * arma::cube, which don't support non-contiguous subviews (so we can't do
 * input.cols(order)).
 */
template<typename InputType>
void SplitHelper(const InputType& input,
                 InputType& train,
                 InputType& test,
                 const double testRatio,
                 const arma::uvec& order = arma::uvec(),
                 const typename std::enable_if_t<
                     IsCube<InputType>::value || IsField<InputType>::value
                 >* = 0)
{
  const size_t testSize = static_cast<size_t>(input.n_cols * testRatio);
  const size_t trainSize = input.n_cols - testSize;

  // Cubes and fields can be initialized with three dimensions.
  train.set_size(input.n_rows, trainSize, input.n_slices);
  test.set_size(input.n_rows, testSize, input.n_slices);

  // Shuffling and splitting simultaneously.
  if (!order.is_empty())
  {
    if (trainSize > 0)
    {
      for (size_t i = 0; i < trainSize; ++i)
        train.col(i) = input.col(order[i]);
    }

    if (trainSize < input.n_cols)
    {
      for (size_t i = trainSize; i < input.n_cols; ++i)
        test.col(i - trainSize) = input.col(order[i]);
    }
  }
  // Splitting only.
  else
  {
    if (trainSize > 0)
      train = input.cols(0, trainSize - 1);

    if (trainSize < input.n_cols)
      test = input.cols(trainSize, input.n_cols - 1);
  }
}

/**
 * Given an input dataset and labels, stratify into a training set and test set.
 * It is recommended to have the input labels between the range [0, n) where n
 * is the number of different labels. The NormalizeLabels() function in
 * mlpack::data can be used for this.
 * Expects labels to be of type arma::Row<> or arma::Col<>.
 * Throws a runtime error if this is not the case.
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
 * RandomSeed(100); // Set the seed if you like.
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
template<typename T, typename LabelsType,
         typename = std::enable_if_t<arma::is_arma_type<LabelsType>::value> >
void StratifiedSplit(const arma::Mat<T>& input,
                     const LabelsType& inputLabel,
                     arma::Mat<T>& trainData,
                     arma::Mat<T>& testData,
                     LabelsType& trainLabel,
                     LabelsType& testLabel,
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
  const bool typeCheck = (arma::is_Row<LabelsType>::value)
      || (arma::is_Col<LabelsType>::value);
  if (!typeCheck)
    throw std::runtime_error("data::Split(): when stratified sampling is done, "
        "labels must have type `arma::Row<>`!");
  util::CheckSameSizes(input, inputLabel, "data::Split()");
  size_t trainIdx = 0;
  size_t testIdx = 0;
  size_t trainSize = 0;
  size_t testSize = 0;
  arma::uvec labelCounts;
  arma::uvec testLabelCounts;
  typename LabelsType::elem_type maxLabel = inputLabel.max();

  labelCounts.zeros(maxLabel+1);
  testLabelCounts.zeros(maxLabel+1);

  for (typename LabelsType::elem_type label : inputLabel)
    ++labelCounts[label];

  for (arma::uword labelCount : labelCounts)
  {
    testSize += floor(labelCount * testRatio);
    trainSize += labelCount - floor(labelCount * testRatio);
  }

  trainData.set_size(input.n_rows, trainSize);
  testData.set_size(input.n_rows, testSize);
  trainLabel.set_size(inputLabel.n_rows, trainSize);
  testLabel.set_size(inputLabel.n_rows, testSize);

  if (shuffleData)
  {
    arma::uvec order = arma::shuffle(
        arma::linspace<arma::uvec>(0, input.n_cols - 1, input.n_cols));

    for (arma::uword i : order)
    {
      typename LabelsType::elem_type label = inputLabel[i];
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
  else
  {
    for (arma::uword i = 0; i < input.n_cols; i++)
    {
      typename LabelsType::elem_type label = inputLabel[i];
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
 * RandomSeed(100); // Set the seed if you like.
 *
 * // Split the dataset into a training and test set, with 30% of the data being
 * // held out for the test set.
 * Split(input, label, trainData,
 *                testData, trainLabel, testLabel, 0.3);
 * @endcode
 *
 * @tparam T Type of the elements of the input matrix.
 * @tparam LabelsType Type of input labels. It can be arma::Mat, arma::Row,
 *       arma::Cube or arma::SpMat.
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
template<typename MatType, typename LabelsType,
         typename = std::enable_if_t<IsAnyArmaBaseType<MatType>::value>,
         typename = std::enable_if_t<IsAnyArmaBaseType<LabelsType>::value>>
void Split(const MatType& input,
           const LabelsType& inputLabel,
           MatType& trainData,
           MatType& testData,
           LabelsType& trainLabel,
           LabelsType& testLabel,
           const double testRatio,
           const bool shuffleData = true)
{
  util::CheckSameSizes(input, inputLabel, "data::Split()");
  if (shuffleData)
  {
    arma::uvec order = arma::shuffle(arma::linspace<arma::uvec>(0,
        input.n_cols - 1, input.n_cols));
    SplitHelper(input, trainData, testData, testRatio, order);
    SplitHelper(inputLabel, trainLabel, testLabel, testRatio, order);
  }
  else
  {
    SplitHelper(input, trainData, testData, testRatio);
    SplitHelper(inputLabel, trainLabel, testLabel, testRatio);
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
 * RandomSeed(100); // Set the seed if you like.
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
template<typename MatType,
         typename = std::enable_if_t<IsAnyArmaBaseType<MatType>::value>>
void Split(const MatType& input,
           MatType& trainData,
           MatType& testData,
           const double testRatio,
           const bool shuffleData = true)
{
  if (shuffleData)
  {
    arma::uvec order = arma::shuffle(arma::linspace<arma::uvec>(0,
        input.n_cols - 1, input.n_cols));
    SplitHelper(input, trainData, testData, testRatio, order);
  }
  else
  {
    SplitHelper(input, trainData, testData, testRatio);
  }
}

/**
 * Given an input dataset, labels, and weights, split into a training set and
 * test set.  Example usage below.  This overload places the split dataset into
 * the six output parameters given (trainData, testData, trainLabels,
 * testLabels, trainWeights, and testWeights).
 *
 * @code
 * arma::mat input = loadData();
 * arma::Row<size_t> label = loadLabel();
 * arma::rowvec weights = loadWeights();
 *
 * arma::mat trainData, testData;
 * arma::Row<size_t> trainLabels, testLabels;
 * arma::rowvec trainWeights, testWeights;
 *
 * // Split the dataset into a training and test set, with 30% of the data being
 * // held out for the test set.
 * Split(input, label, weights, trainData, testData, trainLabels, testLabels,
 *     trainWeights, testWeights, 0.3);
 * @endcode
 *
 * @tparam MatType Type of the data matrix.
 * @tparam LabelsType Type of input labels. It can be arma::Mat, arma::Row,
 *      arma::Cube, or arma::SpMat.
 * @tparam WeightsType Type of input weights.  It can be arma::Mat, arma::Row,
 *      arma::Cube, or arma::SpMat.
 * @param input Input dataset to split.
 * @param inputLabels Input labels to split.
 * @param inputWeights Input weights to split.
 * @param trainData Matrix to store training data into.
 * @param testData Matrix to store test data into.
 * @param trainLabels Vector to store training labels into.
 * @param testLabels Vector to store test labels into.
 * @param trainWeights Vector to store training weights into.
 * @param testWeights Vector to store test weights into.
 * @param testRatio Percentage of dataset to use for test set (between 0 and 1).
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *       sample is visited in linear order. (Default true.)
 */
template<typename MatType, typename LabelsType, typename WeightsType,
         typename = std::enable_if_t<IsAnyArmaBaseType<MatType>::value>,
         typename = std::enable_if_t<IsAnyArmaBaseType<LabelsType>::value>,
         typename = std::enable_if_t<IsAnyArmaBaseType<WeightsType>::value>>
void Split(const MatType& input,
           const LabelsType& inputLabels,
           const WeightsType& inputWeights,
           MatType& trainData,
           MatType& testData,
           LabelsType& trainLabels,
           LabelsType& testLabels,
           WeightsType& trainWeights,
           WeightsType& testWeights,
           const double testRatio,
           const bool shuffleData = true)
{
  util::CheckSameSizes(input, inputLabels, "data::Split()");
  util::CheckSameSizes(input, inputWeights, "data::Split()");
  if (shuffleData)
  {
    arma::uvec order = arma::shuffle(arma::linspace<arma::uvec>(0,
        input.n_cols - 1, input.n_cols));
    SplitHelper(input, trainData, testData, testRatio, order);
    SplitHelper(inputLabels, trainLabels, testLabels, testRatio, order);
    SplitHelper(inputWeights, trainWeights, testWeights, testRatio, order);
  }
  else
  {
    SplitHelper(input, trainData, testData, testRatio);
    SplitHelper(inputLabels, trainLabels, testLabels, testRatio);
    SplitHelper(inputWeights, trainWeights, testWeights, testRatio);
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
 * @tparam T Type of the elements of the input matrix.
 * @tparam LabelsType Type of input labels. It can be arma::Mat, arma::Row,
 *       arma::Cube or arma::SpMat.
 * @param input Input dataset to split.
 * @param inputLabel Input labels to split.
 * @param testRatio Percentage of dataset to use for test set (between 0 and 1).
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *     sample is visited in linear order. (Default true).
 * @param stratifyData If true, the train and test splits are stratified
 *     so that the ratio of each class in the training and test sets is the same
 *     as in the original dataset. Expects labels to be of type arma::Row<> or
 *     arma::Col<>.
 * @return std::tuple containing trainData (arma::Mat<T>), testData
 *      (arma::Mat<T>), trainLabel (arma::Row<U>), and testLabel (arma::Row<U>).
 */
template<typename T, typename LabelsType,
         typename = std::enable_if_t<arma::is_arma_type<LabelsType>::value> >
[[deprecated("Will be removed in mlpack 5.0.0; use other overloads instead")]]
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
[[deprecated("Will be removed in mlpack 5.0.0; use other overloads instead")]]
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

/**
 * Given an input dataset, split into a training set and test set.
 * Example usage below. This overload places the split dataset into the two
 * output parameters given (trainData, testData).
 *
 * @code
 * arma::field<arma::mat> input = loadData();
 * arma::field<arma::mat> trainData;
 * arma::field<arma::mat> testData;
 * RandomSeed(100); // Set the seed if you like.
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
void Split(const arma::field<T>& input,
           arma::field<T>& trainData,
           arma::field<T>& testData,
           const double testRatio,
           const bool shuffleData = true)
{
  if (shuffleData)
  {
    arma::uvec order = arma::shuffle(arma::linspace<arma::uvec>(0,
        input.n_cols - 1, input.n_cols));
    SplitHelper(input, trainData, testData, testRatio, order);
  }
  else
  {
    SplitHelper(input, trainData, testData, testRatio);
  }
}

/**
 * Given an input dataset and labels, split into a training set and test set.
 * Example usage below.  This overload places the split dataset into the four
 * output parameters given (trainData, testData, trainLabel, and testLabel).
 *
 * The input dataset must be of type arma::field. It should have the shape -
 * (n_rows = 1, n_cols = Number of samples, n_slices = 1).
 *
 * NOTE: Here FieldType could be arma::field<arma::mat> or
 * arma::field<arma::vec>.
 *
 * @code
 * arma::field<arma::mat> input = loadData();
 * arma::field<arma::vec> label = loadLabel();
 * arma::field<arma::mat> trainData;
 * arma::field<arma::mat> testData;
 * arma::field<arma::vec> trainLabel;
 * arma::field<arma::vec> testLabel;
 * RandomSeed(100); // Set the seed if you like.
 *
 * // Split the dataset into a training and test set, with 30% of the data being
 * // held out for the test set.
 * Split(input, label, trainData, trainLabel, testData, testLabel, 0.3);
 * @endcode
 *
 * @param input Input dataset to split.
 * @param inputLabel Input labels to split.
 * @param trainData FieldType to store training data into.
 * @param testData FieldType test data into.
 * @param trainLabel Field vector to store training labels into.
 * @param testLabel Field vector to store test labels into.
 * @param testRatio Percentage of dataset to use for test set (between 0 and 1).
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *       sample is visited in linear order. (Default true.)
 */
template <typename FieldType, typename T,
          typename = std::enable_if_t<
              arma::is_Col<typename FieldType::object_type>::value ||
              arma::is_Mat_only<typename FieldType::object_type>::value>>
[[deprecated("Use other field overload with testData before trainLabel; this "
             "overload will be removed in mlpack 5.0.0.")]]
void Split(const FieldType& input,
           const arma::field<T>& inputLabel,
           FieldType& trainData,
           arma::field<T>& trainLabel,
           FieldType& testData,
           arma::field<T>& testLabel,
           const double testRatio,
           const bool shuffleData = true)
{
  util::CheckSameSizes(input, inputLabel, "data::Split()");
  if (shuffleData)
  {
    arma::uvec order = arma::shuffle(arma::linspace<arma::uvec>(0,
        input.n_cols - 1, input.n_cols));
    SplitHelper(input, trainData, testData, testRatio, order);
    SplitHelper(inputLabel, trainLabel, testLabel, testRatio, order);
  }
  else
  {
    SplitHelper(input, trainData, testData, testRatio);
    SplitHelper(inputLabel, trainLabel, testLabel, testRatio);
  }
}

/**
 * Given an input dataset and labels, split into a training set and test set.
 * Example usage below.  This overload returns the split dataset as a std::tuple
 * with four elements: an FieldType containing the training data, an
 * FieldType containing the test data, an arma::field<arma::vec> containing the
 * training labels, and an arma::field<arma::vec> containing the test labels.
 *
 * The input dataset must be of type arma::field. It should have the shape -
 * (n_rows = 1, n_cols = Number of samples, n_slices = 1)
 *
 * NOTE: Here FieldType could be arma::field<arma::mat> or arma::field<arma::vec>
 *
 * @code
 * arma::field<arma::mat> input = loadData();
 * arma::field<arma::vec> label = loadLabel();
 * auto splitResult = Split(input, label, 0.2);
 * @endcode
 *
 * @param input Input dataset to split.
 * @param inputLabel Input labels to split.
 * @param testRatio Percentage of dataset to use for test set (between 0 and 1).
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *       sample is visited in linear order. (Default true).
 * @return std::tuple containing trainData (FieldType), testData
 *      (FieldType), trainLabel (arma::field<arma::vec>), and
 *                   testLabel (arma::field<arma::vec>).
 */
template <class FieldType, typename T,
          class = std::enable_if_t<
              arma::is_Col<typename FieldType::object_type>::value ||
              arma::is_Mat_only<typename FieldType::object_type>::value>>
[[deprecated("Will be removed in mlpack 5.0.0; use other overloads instead")]]
std::tuple<FieldType, FieldType, arma::field<T>, arma::field<T>>
Split(const FieldType& input,
      const arma::field<T>& inputLabel,
      const double testRatio,
      const bool shuffleData = true)
{
  FieldType trainData;
  FieldType testData;
  arma::field<T> trainLabel;
  arma::field<T> testLabel;

  Split(input, inputLabel, trainData, trainLabel, testData, testLabel,
      testRatio, shuffleData);

  return std::make_tuple(std::move(trainData),
                         std::move(testData),
                         std::move(trainLabel),
                         std::move(testLabel));
}

/**
 * Given an input dataset, split into a training set and test set.
 * Example usage below.  This overload returns the split dataset as a std::tuple
 * with two elements: an FieldType containing the training data and an
 * FieldType containing the test data.
 *
 * The input dataset must be of type arma::field. It should have the shape -
 * (n_rows = 1, n_cols = Number of samples, n_slices = 1)
 *
 * NOTE: Here FieldType could be arma::field<arma::mat> or arma::field<arma::vec>
 *
 * @code
 * arma::field<arma::mat> input = loadData();
 * auto splitResult = Split(input, 0.2);
 * @endcode
 *
 * @param input Input dataset to split.
 * @param testRatio Percentage of dataset to use for test set (between 0 and 1).
 * @param shuffleData If true, the sample order is shuffled; otherwise, each
 *       sample is visited in linear order. (Default true).
 * @return std::tuple containing trainData (FieldType)
 *      and testData (FieldType).
 */
template <class FieldType,
          class = std::enable_if_t<
              arma::is_Col<typename FieldType::object_type>::value ||
              arma::is_Mat_only<typename FieldType::object_type>::value>>
[[deprecated("Will be removed in mlpack 5.0.0; use other overloads instead")]]
std::tuple<FieldType, FieldType>
Split(const FieldType& input,
      const double testRatio,
      const bool shuffleData = true)
{
  FieldType trainData;
  FieldType testData;
  Split(input, trainData, testData, testRatio, shuffleData);

  return std::make_tuple(std::move(trainData),
                         std::move(testData));
}

} // namespace data
} // namespace mlpack

#endif
