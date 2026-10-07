/**
 * @file methods/sefr/sefr.hpp
 * @author Hamidreza Keshavarz
 *
 * Definition of the SEFR (Scalable, Efficient, and Fast classifieR) class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SEFR_SEFR_HPP
#define MLPACK_METHODS_SEFR_SEFR_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * An implementation of SEFR, a linear classifier with linear-time training and
 * no hyperparameters.  For each class, SEFR computes the mean of the points in
 * that class and the mean of the points outside of it; each feature's weight is
 * the normalized difference of those two means, and the bias is a weighted
 * average of the mean scores of the two groups.  Multiclass problems are
 * handled one-vs-all.
 *
 * The weight formula assumes non-negative features.  To accept any data, each
 * feature is shifted by its minimum over the training data when that minimum
 * is negative (see Offsets()); the shift is folded into the biases, so it costs
 * nothing at prediction time, and non-negative data is unaffected.
 *
 * Only per-class sums of points and per-class counts are stored, so the model
 * can be updated incrementally one point at a time, and training with instance
 * weights is supported.
 *
 * For more information, see the following paper:
 *
 * @code
 * @article{keshavarz2020sefr,
 *   title = {{SEFR}: A Fast Linear-Time Classifier for Ultra-Low Power
 *       Devices},
 *   author = {Keshavarz, Hamidreza and Abadeh, Mohammad Saniee and
 *       Rawassizadeh, Reza},
 *   journal = {arXiv preprint arXiv:2006.04620},
 *   year = {2020}
 * }
 * @endcode
 *
 * @tparam ModelMatType Matrix type used to store the model parameters.
 */
template<typename ModelMatType = arma::mat>
class SEFR
{
 public:
  //! The element type used to store the model.
  using ElemType = typename ModelMatType::elem_type;
  //! The dense matrix type used to store the model.
  using DenseMatType = typename GetDenseMatType<ModelMatType>::type;
  //! The dense column vector type used to store the model.
  using DenseColType = typename GetDenseColType<ModelMatType>::type;

  /**
   * Create an untrained SEFR model with the given number of classes and
   * dimensionality.  All weights and biases are zero.
   *
   * @param numClasses Number of classes.
   * @param dimensionality Dimensionality of the data.
   */
  SEFR(const size_t numClasses = 0, const size_t dimensionality = 0);

  /**
   * Train a SEFR model on the given data and labels.  Labels must be in the
   * range [0, numClasses).
   *
   * @param data Training data (one point per column).
   * @param labels Labels for each point.
   * @param numClasses Number of classes.
   */
  template<typename MatType>
  SEFR(const MatType& data,
       const arma::Row<size_t>& labels,
       const size_t numClasses);

  /**
   * Train a SEFR model on the given data, labels, and instance weights.
   * Labels must be in the range [0, numClasses).
   *
   * @param data Training data (one point per column).
   * @param labels Labels for each point.
   * @param numClasses Number of classes.
   * @param instanceWeights Weight of each training point.
   */
  template<typename MatType, typename WeightsType>
  SEFR(const MatType& data,
       const arma::Row<size_t>& labels,
       const size_t numClasses,
       const WeightsType& instanceWeights,
       const std::enable_if_t<
           arma::is_arma_type<WeightsType>::value>* = 0);

  /**
   * Train the model on the given data and labels.  Any previously trained
   * model is discarded.
   *
   * @param data Training data (one point per column).
   * @param labels Labels for each point.
   * @param numClasses Number of classes.
   */
  template<typename MatType>
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses);

  /**
   * Train the model on the given data, labels, and instance weights.  Any
   * previously trained model is discarded.
   *
   * @param data Training data (one point per column).
   * @param labels Labels for each point.
   * @param numClasses Number of classes.
   * @param instanceWeights Weight of each training point.
   */
  template<typename MatType, typename WeightsType>
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const WeightsType& instanceWeights,
             const std::enable_if_t<
                 arma::is_arma_type<WeightsType>::value>* = 0);

  /**
   * Incrementally update the model with a single point.  The model must have
   * been constructed or trained with the correct number of classes and
   * dimensionality.  The result is identical to batch training on all points
   * seen so far.
   *
   * @param point Point to train on.
   * @param label Label of the point.
   */
  template<typename VecType>
  void Train(const VecType& point, const size_t label);

  /**
   * Classify the given point.
   *
   * @param point Point to classify.
   * @return Predicted label of the point.
   */
  template<typename VecType>
  size_t Classify(const VecType& point) const;

  /**
   * Classify the given point, also returning the score of each class.  Scores
   * are not probabilities and may take any value.
   *
   * @param point Point to classify.
   * @param label Predicted label of the point.
   * @param scores Score of each class for the point.
   */
  template<typename VecType>
  void Classify(const VecType& point,
                size_t& label,
                DenseColType& scores) const;

  /**
   * Classify the given points.
   *
   * @param data Points to classify (one point per column).
   * @param labels Predicted label of each point.
   */
  template<typename MatType>
  void Classify(const MatType& data, arma::Row<size_t>& labels) const;

  /**
   * Classify the given points, also returning the score of each class for
   * each point.  Scores are not probabilities and may take any value.
   *
   * @param data Points to classify (one point per column).
   * @param labels Predicted label of each point.
   * @param scores Score of each class for each point (one column per point).
   */
  template<typename MatType>
  void Classify(const MatType& data,
                arma::Row<size_t>& labels,
                DenseMatType& scores) const;

  /**
   * Reset the model to the untrained state, keeping the number of classes and
   * dimensionality.
   */
  void Reset();

  //! Get the number of classes.
  size_t NumClasses() const { return weights.n_cols; }
  //! Get the dimensionality of the model.
  size_t Dimensionality() const { return weights.n_rows; }

  //! Get the weights (one column per class).
  const DenseMatType& Weights() const { return weights; }
  //! Get the biases (one element per class).
  const DenseColType& Biases() const { return biases; }

  //! Get the sum of the (weighted) training points of each class.
  const DenseMatType& ClassSums() const { return classSums; }
  //! Get the (weighted) number of training points of each class.
  const DenseColType& ClassCounts() const { return classCounts; }
  //! Get the per-feature offsets subtracted from the data before computing
  //! the weights: the minimum of each feature over the training data, or 0
  //! when that minimum is non-negative.
  const DenseColType& Offsets() const { return offsets; }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Check that labels are valid and sizes match.
  template<typename MatType>
  void CheckTrainingData(const MatType& data,
                         const arma::Row<size_t>& labels,
                         const size_t numClasses) const;

  //! Set the offsets from the minimum of each feature of the given data.
  template<typename MatType>
  void ComputeOffsets(const MatType& data);

  //! Accumulate class sums and counts from the given data.
  template<typename MatType, typename WeightsType>
  void Accumulate(const MatType& data,
                  const arma::Row<size_t>& labels,
                  const WeightsType& instanceWeights);

  //! Compute weights and biases from the class sums and counts.
  void ComputeModel();

  //! Weights; each column corresponds to a class.
  DenseMatType weights;
  //! Biases; one per class.
  DenseColType biases;
  //! Sum of the (weighted) training points of each class.
  DenseMatType classSums;
  //! (Weighted) number of training points of each class.
  DenseColType classCounts;
  //! Per-feature offsets: min(0, minimum of the feature over training data).
  DenseColType offsets;
};

} // namespace mlpack

// Include implementation.
#include "sefr_impl.hpp"

#endif
