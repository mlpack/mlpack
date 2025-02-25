/**
 * @file core/cv/k_fold_cv.hpp
 * @author Kirill Mishchenko
 *
 * k-fold cross-validation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_K_FOLD_CV_HPP
#define MLPACK_CORE_CV_K_FOLD_CV_HPP

#include <mlpack/core/cv/meta_info_extractor.hpp>
#include <mlpack/core/cv/cv_base.hpp>

namespace mlpack {

/**
 * The class KFoldCV implements k-fold cross-validation for regression and
 * classification algorithms.
 *
 * To construct a KFoldCV object you need to pass the k parameter and arguments
 * that specify data. For example, you can run 10-fold cross-validation for
 * SoftmaxRegression in the following way.
 *
 * @code
 * // 100-point 5-dimensional random dataset.
 * arma::mat data = arma::randu<arma::mat>(5, 100);
 * // Random labels in the [0, 4] interval.
 * arma::Row<size_t> labels =
 *     arma::randi<arma::Row<size_t>>(100, arma::distr_param(0, 4));
 * size_t numClasses = 5;
 *
 * KFoldCV<SoftmaxRegression<>, Accuracy> cv(10, data, labels, numClasses);
 *
 * double lambda = 0.1;
 * double softmaxAccuracy = cv.Evaluate(lambda);
 * @endcode
 *
 * Before calling @c Evaluate(), it is possible to shuffle the data by calling
 * the @c Shuffle() function.  Shuffling is performed at construction time if
 * the parameter @c shuffle is set to @c true in the constructor.
 *
 * @tparam MLAlgorithm A machine learning algorithm.
 * @tparam Metric A metric to assess the quality of a trained model.
 * @tparam MatType The type of data.
 * @tparam PredictionsType The type of predictions (should be passed when the
 *     predictions type is a template parameter in Train methods of
 *     MLAlgorithm).
 * @tparam WeightsType The type of weights (should be passed when weighted
 *     learning is supported, and the weights type is a template parameter in
 *     Train methods of MLAlgorithm).
 */
template<typename MLAlgorithm,
         typename Metric,
         typename MatType = arma::mat,
         typename PredictionsType =
             typename MetaInfoExtractor<MLAlgorithm, MatType>::PredictionsType,
         typename WeightsType =
             typename MetaInfoExtractor<MLAlgorithm, MatType,
                 PredictionsType>::WeightsType>
class KFoldCV
{
 public:
  /**
   * This constructor can be used for regression algorithms and for binary
   * classification algorithms.
   *
   * @param k Number of folds (should be at least 2).
   * @param xs Data points to cross-validate on.
   * @param ys Predictions (labels for classification algorithms and responses
   *     for regression algorithms) for each data point.
   * @param shuffle Whether or not to shuffle the data during construction.
   */
  KFoldCV(const size_t k,
          const MatType& xs,
          const PredictionsType& ys,
          const bool shuffle = true);

  /**
   * This constructor can be used for multiclass classification algorithms.
   *
   * @param k Number of folds (should be at least 2).
   * @param xs Data points to cross-validate on.
   * @param ys Labels for each data point.
   * @param numClasses Number of classes in the dataset.
   * @param shuffle Whether or not to shuffle the data during construction.
   */
  KFoldCV(const size_t k,
          const MatType& xs,
          const PredictionsType& ys,
          const size_t numClasses,
          const bool shuffle = true);

  /**
   * This constructor can be used for multiclass classification algorithms that
   * can take a data::DatasetInfo parameter.
   *
   * @param k Number of folds (should be at least 2).
   * @param xs Data points to cross-validate on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param ys Labels for each data point.
   * @param numClasses Number of classes in the dataset.
   * @param shuffle Whether or not to shuffle the data during construction.
   */
  KFoldCV(const size_t k,
          const MatType& xs,
          const data::DatasetInfo& datasetInfo,
          const PredictionsType& ys,
          const size_t numClasses,
          const bool shuffle = true);

  /**
   * This constructor can be used for regression and binary classification
   * algorithms that support weighted learning.
   *
   * @param k Number of folds (should be at least 2).
   * @param xs Data points to cross-validate on.
   * @param ys Predictions (labels for classification algorithms and responses
   *     for regression algorithms) for each data point.
   * @param weights Observation weights (for boosting).
   * @param shuffle Whether or not to shuffle the data during construction.
   */
  KFoldCV(const size_t k,
          const MatType& xs,
          const PredictionsType& ys,
          const WeightsType& weights,
          const bool shuffle = true);

  /**
   * This constructor can be used for multiclass classification algorithms that
   * support weighted learning.
   *
   * @param k Number of folds (should be at least 2).
   * @param xs Data points to cross-validate on.
   * @param ys Labels for each data point.
   * @param numClasses Number of classes in the dataset.
   * @param weights Observation weights (for boosting).
   * @param shuffle Whether or not to shuffle the data during construction.
   */
  KFoldCV(const size_t k,
          const MatType& xs,
          const PredictionsType& ys,
          const size_t numClasses,
          const WeightsType& weights,
          const bool shuffle = true);

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
   * @param shuffle Whether or not to shuffle the data during construction.
   */
  KFoldCV(const size_t k,
          const MatType& xs,
          const data::DatasetInfo& datasetInfo,
          const PredictionsType& ys,
          const size_t numClasses,
          const WeightsType& weights,
          const bool shuffle = true);

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

 private:
  //! A short alias for CVBase.
  using Base = CVBase<MLAlgorithm, MatType, PredictionsType, WeightsType>;

 public:
  /**
   * Shuffle the data.  This overload is called if weights are not supported by
   * the model type.
   */
  template<bool Enabled = !Base::MIE::SupportsWeights,
           typename = std::enable_if_t<Enabled>>
  void Shuffle();

  /**
   * Shuffle the data.  This overload is called if weights are supported by the
   * model type.
   */
  template<bool Enabled = Base::MIE::SupportsWeights,
           typename = std::enable_if_t<Enabled>,
           typename = void>
  void Shuffle();

 private:
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
  KFoldCV(Base&& base,
          const size_t k,
          const MatType& xs,
          const PredictionsType& ys,
          const bool shuffle);

  /**
   * Assert the k parameter and data consistency and initialize fields required
   * for running k-fold cross-validation in the case of weighted learning.
   */
  KFoldCV(Base&& base,
          const size_t k,
          const MatType& xs,
          const PredictionsType& ys,
          const WeightsType& weights,
          const bool shuffle);

  /**
   * Initialize the given destination matrix with the given source joined with
   * its first k - 2 bins.
   */
  template<typename DataType>
  void InitKFoldCVMat(const DataType& source, DataType& destination);

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

  /**
   * Calculate the index of the first column of the ith validation subset.
   *
   * We take the ith validation subset after the ith training subset if
   * i < k - 1 and before it otherwise.
   */
  inline size_t ValidationSubsetFirstCol(const size_t i);

  /**
   * Get the ith training subset from a variable of a matrix type.
   */
  template<typename ElementType>
  inline arma::Mat<ElementType> GetTrainingSubset(arma::Mat<ElementType>& m,
                                                  const size_t i);

  /**
   * Get the ith training subset from a variable of a row type.
   */
  template<typename ElementType>
  inline arma::Row<ElementType> GetTrainingSubset(arma::Row<ElementType>& r,
                                                  const size_t i);

  /**
   * Get the ith validation subset from a variable of a matrix type.
   */
  template<typename ElementType>
  inline arma::Mat<ElementType> GetValidationSubset(arma::Mat<ElementType>& m,
                                                    const size_t i);

  /**
   * Get the ith validation subset from a variable of a row type.
   */
  template<typename ElementType>
  inline arma::Row<ElementType> GetValidationSubset(arma::Row<ElementType>& r,
                                                    const size_t i);
};

} // namespace mlpack

// Include implementation
#include "k_fold_cv_impl.hpp"

#endif
