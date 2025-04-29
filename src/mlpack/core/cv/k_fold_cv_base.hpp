/**
 * @file
 * k-fold cross-validation base class.
 * 
 * @authors Kirill Mishchenko, Felix Patschkowski
 *
 * @copyright mlpack is free software; you may redistribute it and/or modify it
 * under the terms of the 3-clause BSD license.  You should have received a copy
 * of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_K_FOLD_CV_BASE_HPP
#define MLPACK_CORE_CV_K_FOLD_CV_BASE_HPP

#include <mlpack/core/cv/meta_info_extractor.hpp>
#include <mlpack/core/cv/cv_base.hpp>

namespace mlpack {

template<typename Derived,
         typename MLAlgorithm,
         typename Metric,
         typename MatType = arma::mat,
         typename PredictionsType =
             typename MetaInfoExtractor<MLAlgorithm,
                                        MatType>::PredictionsType,
         typename WeightsType =
             typename MetaInfoExtractor<MLAlgorithm,
                                        MatType,
                                        PredictionsType>::WeightsType>
class KFoldCVBase
{
 public:
  /**
    * Run k-fold cross-validation.
    *
    * @param args Arguments for MLAlgorithm (in addition to the passed
    *     ones in the constructor).
    */
  template<typename... MLAlgorithmArgs>
  double Evaluate(const MLAlgorithmArgs& ...args);

  //! Access and modify a model from the last run of k-fold cross-validation.
  MLAlgorithm& Model();

 protected:
  /**
    * This constructor can be used for regression algorithms and for binary
    * classification algorithms.
    *
    * @param k Number of folds (should be at least 2).
    * @param xs Data points to cross-validate on.
    * @param ys Predictions (labels for classification algorithms and responses
    *     for regression algorithms) for each data point.
    */
  KFoldCVBase(const size_t k,
              const MatType& xs,
              const PredictionsType& ys);

  /**
    * This constructor can be used for multiclass classification algorithms.
    *
    * @param k Number of folds (should be at least 2).
    * @param xs Data points to cross-validate on.
    * @param ys Labels for each data point.
    * @param numClasses Number of classes in the dataset.
    */
  KFoldCVBase(const size_t k,
              const MatType& xs,
              const PredictionsType& ys,
              const size_t numClasses);

  /**
    * This constructor can be used for multiclass classification algorithms that
    * can take a data::DatasetInfo parameter.
    *
    * @param k Number of folds (should be at least 2).
    * @param xs Data points to cross-validate on.
    * @param datasetInfo Type information for each dimension of the dataset.
    * @param ys Labels for each data point.
    * @param numClasses Number of classes in the dataset.
    */
  KFoldCVBase(const size_t k,
              const MatType& xs,
              const data::DatasetInfo& datasetInfo,
              const PredictionsType& ys,
              const size_t numClasses);

  /**
    * This constructor can be used for regression and binary classification
    * algorithms that support weighted learning.
    *
    * @param k Number of folds (should be at least 2).
    * @param xs Data points to cross-validate on.
    * @param ys Predictions (labels for classification algorithms and responses
    *     for regression algorithms) for each data point.
    * @param weights Observation weights (for boosting).
    */
  KFoldCVBase(const size_t k,
              const MatType& xs,
              const PredictionsType& ys,
              const WeightsType& weights);

  /**
    * This constructor can be used for multiclass classification algorithms that
    * support weighted learning.
    *
    * @param k Number of folds (should be at least 2).
    * @param xs Data points to cross-validate on.
    * @param ys Labels for each data point.
    * @param numClasses Number of classes in the dataset.
    * @param weights Observation weights (for boosting).
    */
  KFoldCVBase(const size_t k,
              const MatType& xs,
              const PredictionsType& ys,
              const size_t numClasses,
              const WeightsType& weights);

  /**
    * This constructor can be used for multiclass classification algorithms that
    * can take a data::DatasetInfo parameter and support weighted learning.
    *
    * @param k Number of folds (should be at least 2).
    * @param xs Data points to cross-validate on.
    * @param datasetInfo Type information for each dimension of the dataset.
    * @param ys Labels for each data point.
    * @param numClasses Number of classes in the dataset.
    * @param weights Observation weights (for boosting).
    */
  KFoldCVBase(const size_t k,
              const MatType& xs,
              const data::DatasetInfo& datasetInfo,
              const PredictionsType& ys,
              const size_t numClasses,
              const WeightsType& weights);

 protected:
  /**
   * Initialize the given destination matrix with the given source joined with
   * its first k - 2 bins.
   */
  template<typename DataType>
  void InitKFoldCVMat(const DataType& source, DataType& destination);

  //! A short alias for CVBase.
  using Base = CVBase<MLAlgorithm, MatType, PredictionsType, WeightsType>;

  //! An auxiliary object.
  Base base;

  //! The number of bins in the dataset.
  const size_t k;

  //! The extended (by repeating the first k - 2 bins) data points.
  MatType xs;
  //! The extended (by repeating the first k - 2 bins) predictions.
  PredictionsType ys;
  //! The extended (by repeating the first k - 2 bins) weights.
  WeightsType weights;

  //! The original size of the dataset.
  size_t lastBinSize;

  //! The size of each bin in terms of data points.
  size_t binSize;

  //! A pointer to a model from the last run of k-fold cross-validation.
  std::unique_ptr<MLAlgorithm> modelPtr;

  /**
    * Assert the k parameter and data consistency and initialize fields required
    * for running k-fold cross-validation.
    */
  KFoldCVBase(Base&& base,
              const size_t k,
              const MatType& xs,
              const PredictionsType& ys);

  /**
    * Assert the k parameter and data consistency and initialize fields required
    * for running k-fold cross-validation in the case of weighted learning.
    */
  KFoldCVBase(Base&& base,
              const size_t k,
              const MatType& xs,
              const PredictionsType& ys,
              const WeightsType& weights);

  /**
    * Train and run evaluation in the case of non-weighted learning.
    */
  template<typename... MLAlgorithmArgs,
           bool Enabled = !Base::MIE::SupportsWeights,
           typename = std::enable_if_t<Enabled>>
  double TrainAndEvaluate(const MLAlgorithmArgs& ...mlAlgorithmArgs);

  /**
    * Train and run evaluation in the case of supporting weighted learning.
    */
  template<typename... MLAlgorithmArgs,
           bool Enabled = Base::MIE::SupportsWeights,
           typename = std::enable_if_t<Enabled>,
           typename = void>
  double TrainAndEvaluate(const MLAlgorithmArgs& ...mlAlgorithmArgs);

  void CheckState() const;
};

} // namespace mlpack

// Include KFoldCVBase
#include "k_fold_cv_base_impl.hpp"

#endif
