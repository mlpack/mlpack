/**
 * @file decision_stump.hpp
 * @author Udit Saxena
 *
 * Definition of decision stumps.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_STUMP_DECISION_STUMP_HPP
#define MLPACK_METHODS_DECISION_STUMP_DECISION_STUMP_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace decision_stump {

/**
 * This class implements a decision stump. It constructs a single level
 * decision tree, i.e., a decision stump. It uses entropy to decide splitting
 * ranges.
 *
 * The stump is parameterized by a splitting dimension (the dimension on which
 * points are split), a vector of bin split values, and a vector of labels for
 * each bin.  Bin i is specified by the range [split[i], split[i + 1]).  The
 * last bin has range up to \infty (split[i + 1] does not exist in that case).
 * Points that are below the first bin will take the label of the first bin.
 *
 * @tparam MatType Type of matrix that is being used (sparse or dense).
 */
template<typename MatType = arma::mat>
class DecisionStump
{
 public:
  /**
   * Constructor. Train on the provided data. Generate a decision stump from
   * data.
   *
   * @param data Input, training data.
   * @param labels Labels of training data.
   * @param classes Number of distinct classes in labels.
   * @param bucketSize Minimum size of bucket when splitting.
   */
  DecisionStump(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t classes,
                const size_t bucketSize = 10);

  /**
   * Alternate constructor which copies the parameters bucketSize and classes
   * from an already initiated decision stump, other. It appropriately sets the
   * weight vector.
   *
   * @param other The other initiated Decision Stump object from
   *      which we copy the values.
   * @param data The data on which to train this object on.
   * @param labels The labels of data.
   * @param weights Weight vector to use while training. For boosting purposes.
   */
  DecisionStump(const DecisionStump<>& other,
                const MatType& data,
                const arma::Row<size_t>& labels,
                const arma::rowvec& weights);

  /**
   * Create a decision stump without training.  This stump will not be useful
   * and will always return a class of 0 for anything that is to be classified,
   * so it would be a prudent idea to call Train() after using this constructor.
   */
  DecisionStump();

  /**
   * Train the decision stump on the given data.  This completely overwrites any
   * previous training data, so after training the stump may be completely
   * different.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each point in the dataset.
   * @param classes Number of classes in the dataset.
   * @param bucketSize Minimum size of bucket when splitting.
   */
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t classes,
             const size_t bucketSize);

  /**
   * Classification function. After training, classify test, and put the
   * predicted classes in predictedLabels.
   *
   * @param test Testing data or data to classify.
   * @param predictedLabels Vector to store the predicted classes after
   *     classifying test data.
   */
  void Classify(const MatType& test, arma::Row<size_t>& predictedLabels);

  //! Access the splitting dimension.
  size_t SplitDimension() const { return splitDimension; }
  //! Modify the splitting dimension (be careful!).
  size_t& SplitDimension() { return splitDimension; }

  //! Access the splitting values.
  const arma::vec& Split() const { return split; }
  //! Modify the splitting values (be careful!).
  arma::vec& Split() { return split; }

  //! Access the labels for each split bin.
  const arma::Col<size_t> BinLabels() const { return binLabels; }
  //! Modify the labels for each split bin (be careful!).
  arma::Col<size_t>& BinLabels() { return binLabels; }

  //! Serialize the decision stump.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! The number of classes (we must store this for boosting).
  size_t classes;
  //! The minimum number of points in a bucket.
  size_t bucketSize;

  //! Stores the value of the dimension on which to split.
  size_t splitDimension;
  //! Stores the splitting values after training.
  arma::vec split;
  //! Stores the labels for each splitting bin.
  arma::Col<size_t> binLabels;

  /**
   * Sets up dimension as if it were splitting on it and finds entropy when
   * splitting on dimension.
   *
   * @param dimension A row from the training data, which might be a
   *     candidate for the splitting dimension.
   * @tparam UseWeights Whether we need to run a weighted Decision Stump.
   */
  template<bool UseWeights>
  double SetupSplitDimension(const arma::rowvec& dimension,
                             const arma::Row<size_t>& labels,
                             const arma::rowvec& weightD);

  /**
   * After having decided the dimension on which to split, train on that
   * dimension.
   *
   * @tparam dimension dimension is the dimension decided by the constructor
   *      on which we now train the decision stump.
   */
  template<typename VecType>
  void TrainOnDim(const VecType& dimension,
                  const arma::Row<size_t>& labels);

  /**
   * After the "split" matrix has been set up, merge ranges with identical class
   * labels.
   */
  void MergeRanges();

  /**
   * Count the most frequently occurring element in subCols.
   *
   * @param subCols The vector in which to find the most frequently occurring
   *      element.
   */
  template<typename VecType>
  double CountMostFreq(const VecType& subCols);

  /**
   * Returns 1 if all the values of featureRow are not same.
   *
   * @param featureRow The dimension which is checked for identical values.
   */
  template<typename VecType>
  int IsDistinct(const VecType& featureRow);

  /**
   * Calculate the entropy of the given dimension.
   *
   * @param labels Corresponding labels of the dimension.
   * @param classes Number of classes.
   * @param weights Weights for this set of labels.
   * @tparam UseWeights If true, the weights in the weight vector will be used
   *      (otherwise they are ignored).
   */
  template<bool UseWeights, typename VecType, typename WeightVecType>
  double CalculateEntropy(const VecType& labels,
                          const WeightVecType& weights);

  /**
   * Train the decision stump on the given data and labels.
   *
   * @param data Dataset to train on.
   * @param labels Labels for dataset.
   * @param weights Weights for this set of labels.
   * @tparam UseWeights If true, the weights in the weight vector will be used
   *      (otherwise they are ignored).
   */
  template<bool UseWeights>
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const arma::rowvec& weights);
};

} // namespace decision_stump
} // namespace mlpack

#include "decision_stump_impl.hpp"

#endif
