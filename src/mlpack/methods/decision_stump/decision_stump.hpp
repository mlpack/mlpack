/**
 * @file decision_stump.hpp
 * @author Udit Saxena
 *
 * Definition of decision stumps.
 */
#ifndef __MLPACK_METHODS_DECISION_STUMP_DECISION_STUMP_HPP
#define __MLPACK_METHODS_DECISION_STUMP_DECISION_STUMP_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace decision_stump {

/**
 * This class implements a decision stump. It constructs a single level
 * decision tree, i.e., a decision stump. It uses entropy to decide splitting
 * ranges.
 *
 * The stump is parameterized by a splitting attribute (the dimension on which
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
                const size_t bucketSize);

  /**
   * Alternate constructor which copies parameters bucketSize and numClass from
   * an already initiated decision stump, other. It appropriately sets the
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

  //! Access the splitting attribute.
  size_t SplitAttribute() const { return splitAttribute; }
  //! Modify the splitting attribute (be careful!).
  size_t& SplitAttribute() { return splitAttribute; }

  //! Access the splitting values.
  const arma::vec& Split() const { return split; }
  //! Modify the splitting values (be careful!).
  arma::vec& Split() { return split; }

  //! Access the labels for each split bin.
  const arma::Col<size_t> BinLabels() const { return binLabels; }
  //! Modify the labels for each split bin (be careful!).
  arma::Col<size_t>& BinLabels() { return binLabels; }

 private:
  //! The number of classes (we must store this for boosting).
  size_t classes;
  //! The minimum number of points in a bucket.
  size_t bucketSize;

  //! Stores the value of the attribute on which to split.
  size_t splitAttribute;
  //! Stores the splitting values after training.
  arma::vec split;
  //! Stores the labels for each splitting bin.
  arma::Col<size_t> binLabels;

  /**
   * Sets up attribute as if it were splitting on it and finds entropy when
   * splitting on attribute.
   *
   * @param attribute A row from the training data, which might be a
   *     candidate for the splitting attribute.
   * @param UseWeights Whether we need to run a weighted Decision Stump.
   */
  template<bool UseWeights>
  double SetupSplitAttribute(const arma::rowvec& attribute,
                             const arma::Row<size_t>& labels,
                             const arma::rowvec& weightD);

  /**
   * After having decided the attribute on which to split, train on that
   * attribute.
   *
   * @param attribute attribute is the attribute decided by the constructor
   *      on which we now train the decision stump.
   */
  void TrainOnAtt(const arma::rowvec& attribute,
                  const arma::Row<size_t>& labels);

  /**
   * After the "split" matrix has been set up, merge ranges with identical class
   * labels.
   */
  void MergeRanges();

  /**
   * Count the most frequently occurring element in subCols.
   *
   * @param subCols The vector in which to find the most frequently
   *     occurring element.
   */
  template<typename VecType>
  double CountMostFreq(const VecType& subCols);

  /**
   * Returns 1 if all the values of featureRow are not same.
   *
   * @param featureRow The attribute which is checked for identical values.
   */
  template<typename VecType>
  int IsDistinct(const VecType& featureRow);

  /**
   * Calculate the entropy of the given attribute.
   *
   * @param labels Corresponding labels of the attribute.
   * @param classes Number of classes.
   * @param weights Weights for this set of labels.
   */
  template<bool UseWeights, typename VecType, typename WeightVecType>
  double CalculateEntropy(const VecType& labels,
                          const WeightVecType& weights);

  /**
   * Train the decision stump on the given data and labels.
   *
   * @param data Dataset to train on.
   * @param labels Labels for dataset.
   * @param UseWeights Whether we need to run a weighted Decision Stump.
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
