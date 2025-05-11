/**
 * @file core/cv/purged_k_fold_cv.hpp
 * @author Felix Patschkowski
 *
 * purged k-fold cross-validation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_PURGED_K_FOLD_CV_HPP
#define MLPACK_CORE_CV_PURGED_K_FOLD_CV_HPP

#include "k_fold_cv_base.hpp"

namespace mlpack {

/**
 * A purged k-fold CV is suitable for time series with informational
 * overlap from the training set into the validation set and vice
 * versa. This CV will remove items from the training set that overlap
 * with the validation set.
 */
template<typename MLAlgorithm,
         typename Metric,
         typename MatType = arma::mat,
         typename PredictionsType =
             typename MetaInfoExtractor<MLAlgorithm, MatType>::PredictionsType,
         typename WeightsType =
             typename MetaInfoExtractor<MLAlgorithm, MatType,
                 PredictionsType>::WeightsType>
class PurgedKFoldCV :
    private KFoldCVBase<
        PurgedKFoldCV<MLAlgorithm,
                      Metric,
                      MatType,
                      PredictionsType,
                      WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>
{
 public:
   using Base = typename KFoldCVBase::Base;
  /**
    * This constructor can be used for regression algorithms and for binary
    * classification algorithms.
    *
    * @param k Number of folds (should be at least 2).
    * @param[in] embargoPercentage Percentage of the dataset size that is used
    *                              to extend the duration of the intervals in
    *                              the validation set to remove elements from
    *                              the training set after the validation set
    *                              (embargo). 
    * @param xs Data points to cross-validate on.
    * @param ys Predictions (labels for classification algorithms and responses
    *     for regression algorithms) for each data point.
    * @param[in] intervals A `2 x xs.n_cols` matrix, where each column stores the
    *                      start and end sample of an interval.
    */
  PurgedKFoldCV(
      size_t k,
      const MatType& xs,
      const PredictionsType& ys,
      double embargoPercentage,
      const MatType& intervals);

  /**
    * This constructor can be used for multiclass classification algorithms.
    *
    * @param k Number of folds (should be at least 2).
    * @param[in] embargoPercentage Percentage of the dataset size that is used
    *                              to extend the duration of the intervals in
    *                              the validation set to remove elements from
    *                              the training set after the validation set
    *                              (embargo). 
    * @param xs Data points to cross-validate on.
    * @param ys Labels for each data point.
    * @param numClasses Number of classes in the dataset.
    * @param[in] intervals A `2 x xs.n_cols` matrix, where each column stores the
    *                      start and end sample of an interval.
    */
  PurgedKFoldCV(
      size_t k,
      const MatType& xs,
      const PredictionsType& ys,
      const size_t numClasses,
      double embargoPercentage,
      const MatType& intervals);

  /**
    * This constructor can be used for multiclass classification algorithms that
    * can take a data::DatasetInfo parameter.
    *
    * @param k Number of folds (should be at least 2).
    * @param[in] embargoPercentage Percentage of the dataset size that is used
    *                              to extend the duration of the intervals in
    *                              the validation set to remove elements from
    *                              the training set after the validation set
    *                              (embargo). 
    * @param xs Data points to cross-validate on.
    * @param datasetInfo Type information for each dimension of the dataset.
    * @param ys Labels for each data point.
    * @param numClasses Number of classes in the dataset.
    * @param[in] intervals A `2 x xs.n_cols` matrix, where each column stores the
    *                      start and end sample of an interval.
    */
  PurgedKFoldCV(
      size_t k,
      const MatType& xs,
      const data::DatasetInfo& datasetInfo,
      const PredictionsType& ys,
      const size_t numClasses,
      double embargoPercentage,
      const MatType& intervals);

  /**
    * This constructor can be used for regression and binary classification
    * algorithms that support weighted learning.
    *
    * @param k Number of folds (should be at least 2).
    * @param[in] embargoPercentage Percentage of the dataset size that is used
    *                              to extend the duration of the intervals in
    *                              the validation set to remove elements from
    *                              the training set after the validation set
    *                              (embargo). 
    * @param xs Data points to cross-validate on.
    * @param ys Predictions (labels for classification algorithms and responses
    *     for regression algorithms) for each data point.
    * @param weights Observation weights (for boosting).
    * @param[in] intervals A `2 x xs.n_cols` matrix, where each column stores the
    *                      start and end sample of an interval.
    */
  PurgedKFoldCV(
      size_t k,
      const MatType& xs,
      const PredictionsType& ys,
      const WeightsType& weights,
      double embargoPercentage,
      const MatType& intervals);

  /**
    * This constructor can be used for multiclass classification algorithms that
    * support weighted learning.
    *
    * @param k Number of folds (should be at least 2).
    * @param[in] embargoPercentage Percentage of the dataset size that is used
    *                              to extend the duration of the intervals in
    *                              the validation set to remove elements from
    *                              the training set after the validation set
    *                              (embargo). 
    * @param xs Data points to cross-validate on.
    * @param ys Labels for each data point.
    * @param numClasses Number of classes in the dataset.
    * @param weights Observation weights (for boosting).
    * @param[in] intervals A `2 x xs.n_cols` matrix, where each column stores the
    *                      start and end sample of an interval.
    */
  PurgedKFoldCV(
      size_t k,
      const MatType& xs,
      const PredictionsType& ys,
      const size_t numClasses,
      const WeightsType& weights,
      double embargoPercentage,
      const MatType& intervals);

  /**
    * This constructor can be used for multiclass classification algorithms that
    * can take a data::DatasetInfo parameter and support weighted learning.
    *
    * @param k Number of folds (should be at least 2).
    * @param[in] embargoPercentage Percentage of the dataset size that is used
    *                              to extend the duration of the intervals in
    *                              the validation set to remove elements from
    *                              the training set after the validation set
    *                              (embargo). 
    * @param xs Data points to cross-validate on.
    * @param datasetInfo Type information for each dimension of the dataset.
    * @param ys Labels for each data point.
    * @param numClasses Number of classes in the dataset.
    * @param weights Observation weights (for boosting).
    * @param[in] intervals A `2 x xs.n_cols` matrix, where each column stores
    *                      the start and end sample of an interval.
    */
  PurgedKFoldCV(
      size_t k,
      const MatType& xs,
      const data::DatasetInfo& datasetInfo,
      const PredictionsType& ys,
      const size_t numClasses,
      const WeightsType& weights,
      double embargoPercentage,
      const MatType& intervals);

  using KFoldCVBase<
      PurgedKFoldCV<MLAlgorithm, Metric, MatType, PredictionsType, WeightsType>,
      MLAlgorithm,
      Metric,
      MatType,
      PredictionsType,
      WeightsType>::Evaluate;

  using KFoldCVBase<
      PurgedKFoldCV<MLAlgorithm, Metric, MatType, PredictionsType, WeightsType>,
      MLAlgorithm,
      Metric,
      MatType,
      PredictionsType,
      WeightsType>::Model;

 protected:
  /**
   * Calculate the index of the first column of the ith validation subset.
   *
   * We take the ith validation subset after the ith training subset if
   * i < k - 1 and before it otherwise.
   */
  inline size_t ValidationSubsetFirstCol(size_t i) const;

  /**
   * Get the ith training subset from a variable of a matrix type.
   */
  template<typename ElementType>
  inline arma::Mat<ElementType> GetTrainingSubset(
      const arma::Mat<ElementType>& m,
      size_t i);

  /**
   * Get the ith training subset from a variable of a row type.
   */
  template<typename ElementType>
  inline arma::Row<ElementType> GetTrainingSubset(
      const arma::Row<ElementType>& r,
      size_t i);

  /**
   * Get the ith validation subset from a variable of a matrix type.
   */
  template<typename ElementType>
  inline arma::Mat<ElementType> GetValidationSubset(
      const arma::Mat<ElementType>& m,
      size_t i);

  /**
   * Get the ith validation subset from a variable of a row type.
   */
  template<typename ElementType>
  inline arma::Row<ElementType> GetValidationSubset(
      const arma::Row<ElementType>& r,
      size_t i);

 private:
  template<typename, typename, typename, typename, typename, typename>
  friend class KFoldCVBase;

  /**
   * Check the invariants on the intervals member.
   */
  void CheckIntervals() const;

  /** @brief Get the column indices that form the training subset.
   *
   * @param[in] i The ith training subset to retrieve.
   */
  arma::uvec GetTrainingSubsetCols(size_t i) const;

  /** @brief Get the column indices that form the validation subset.
   *
   * @param[in] i The ith validation subset to retrieve.
   */
  arma::span GetValidationSubsetCols(size_t i) const;

  void CheckState() const;

  const double  embargoPercentage;
  const MatType intervals;
};

} // namespace mlpack

// Include implementation
#include "purged_k_fold_cv_impl.hpp"

#endif
