/**
 * @file core/cv/simple_cv.hpp
 * @author Kirill Mishchenko
 *
 * A simple cross-validation strategy.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_SIMPLE_CV_HPP
#define MLPACK_CORE_CV_SIMPLE_CV_HPP

#include <mlpack/core/cv/meta_info_extractor.hpp>
#include <mlpack/core/cv/cv_base.hpp>

namespace mlpack {

/**
 * SimpleCV splits data into two sets - training and validation sets - and then
 * runs training on the training set and evaluates performance on the validation
 * set.
 *
 * To construct a SimpleCV object you need to pass the validationSize parameter
 * and arguments that specify data. For example, SoftmaxRegression can be
 * validated in the following way.
 *
 * @code
 * // 100-point 5-dimensional random dataset.
 * arma::mat data = arma::randu<arma::mat>(5, 100);
 * // Random labels in the [0, 4] interval.
 * arma::Row<size_t> labels =
 *     arma::randi<arma::Row<size_t>>(100, arma::distr_param(0, 4));
 * size_t numClasses = 5;
 *
 * double validationSize = 0.2;
 * SimpleCV<SoftmaxRegression<>, Accuracy> cv(validationSize, data, labels,
 *     numClasses);
 *
 * double lambda = 0.1;
 * double softmaxAccuracy = cv.Evaluate(lambda);
 * @endcode
 *
 * In the example above, 80% of the passed dataset will be used for training,
 * and remaining 20% will be used for calculating the accuracy metric.
 *
 * @tparam MLAlgorithm A machine learning algorithm.
 * @tparam Metric A metric to assess the quality of a trained model.
 * @tparam MatType The type of data.
 * @tparam PredictionsType The type of predictions (should be passed when the
 *     predictions type is a template parameter in Train methods of the given
 *     MLAlgorithm; arma::Row<size_t> will be used otherwise).
 * @tparam WeightsType The type of weights (should be passed when weighted
 *     learning is supported, and the weights type is a template parameter in
 *     Train methods of the given MLAlgorithm; arma::vec will be used
 *     otherwise).
 */
template<typename MLAlgorithm,
         typename Metric,
         typename MatType = arma::mat,
         typename PredictionsType =
             typename MetaInfoExtractor<MLAlgorithm, MatType>::PredictionsType,
         typename WeightsType =
             typename MetaInfoExtractor<MLAlgorithm, MatType,
                 PredictionsType>::WeightsType>
class SimpleCV
{
 public:
  /**
   * This constructor can be used for regression algorithms and for binary
   * classification algorithms.
   *
   * @param validationSize A proportion (between 0 and 1) of data used as a
   *     validation set.
   * @param xs Data points to cross-validate on.
   * @param ys Predictions (labels for classification algorithms and responses
   *     for regression algorithms) for each data point.
   *
   * @tparam MatInType A type that can be converted to MatType.
   * @tparam PredictionsInType A type that can be converted to PredictionsType.
   */
  template<typename MatInType, typename PredictionsInType>
  SimpleCV(const double validationSize,
           MatInType&& xs,
           PredictionsInType&& ys);

  /**
   * This constructor can be used for multiclass classification algorithms.
   *
   * @param validationSize A proportion (between 0 and 1) of data used as a
   *     validation set.
   * @param xs Data points to cross-validate on.
   * @param ys Labels for each data point.
   * @param numClasses Number of classes in the dataset.
   *
   * @tparam MatInType A type that can be converted to MatType.
   * @tparam PredictionsInType A type that can be converted to PredictionsType.
   */
  template<typename MatInType, typename PredictionsInType>
  SimpleCV(const double validationSize,
           MatInType&& xs,
           PredictionsInType&& ys,
           const size_t numClasses);

  /**
   * This constructor can be used for multiclass classification algorithms that
   * can take a data::DatasetInfo parameter.
   *
   * @param validationSize A proportion (between 0 and 1) of data used as a
   *     validation set.
   * @param xs Data points to cross-validate on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param ys Labels for each data point.
   * @param numClasses Number of classes in the dataset.
   *
   * @tparam MatInType A type that can be converted to MatType.
   * @tparam PredictionsInType A type that can be converted to PredictionsType.
   */
  template<typename MatInType, typename PredictionsInType>
  SimpleCV(const double validationSize,
           MatInType&& xs,
           const data::DatasetInfo& datasetInfo,
           PredictionsInType&& ys,
           const size_t numClasses);

  /**
   * This constructor can be used for regression and binary classification
   * algorithms that support weighted learning.
   *
   * @param validationSize A proportion (between 0 and 1) of data used as a
   *     validation set.
   * @param xs Data points to cross-validate on.
   * @param ys Predictions (labels for classification algorithms and responses
   *     for regression algorithms) for each data point.
   * @param weights Observation weights (for boosting).
   *
   * @tparam MatInType A type that can be converted to MatType.
   * @tparam PredictionsInType A type that can be converted to PredictionsType.
   * @tparam WeightsInType A type that can be converted to WeightsType.
   */
  template<typename MatInType,
           typename PredictionsInType,
           typename WeightsInType>
  SimpleCV(const double validationSize,
           MatInType&& xs,
           PredictionsInType&& ys,
           WeightsInType&& weights);

  /**
   * This constructor can be used for multiclass classification algorithms that
   * support weighted learning.
   *
   * @param validationSize A proportion (between 0 and 1) of data used as a
   *     validation set.
   * @param xs Data points to cross-validate on.
   * @param ys Labels for each data point.
   * @param numClasses Number of classes in the dataset.
   * @param weights Observation weights (for boosting).
   *
   * @tparam MatInType A type that can be converted to MatType.
   * @tparam PredictionsInType A type that can be converted to PredictionsType.
   * @tparam WeightsInType A type that can be converted to WeightsType.
   */
  template<typename MatInType,
           typename PredictionsInType,
           typename WeightsInType>
  SimpleCV(const double validationSize,
           MatInType&& xs,
           PredictionsInType&& ys,
           const size_t numClasses,
           WeightsInType&& weights);

  /**
   * This constructor can be used for multiclass classification algorithms that
   * can take a data::DatasetInfo parameter and support weighted learning.
   *
   * @param validationSize A proportion (between 0 and 1) of data used as a
   *     validation set.
   * @param xs Data points to cross-validate on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param ys Labels for each data point.
   * @param numClasses Number of classes in the dataset.
   * @param weights Observation weights (for boosting).
   *
   * @tparam MatInType A type that can be converted to MatType.
   * @tparam PredictionsInType A type that can be converted to PredictionsType.
   * @tparam WeightsInType A type that can be converted to WeightsType.
   */
  template<typename MatInType,
           typename PredictionsInType,
           typename WeightsInType>
  SimpleCV(const double validationSize,
           MatInType&& xs,
           const data::DatasetInfo& datasetInfo,
           PredictionsInType&& ys,
           const size_t numClasses,
           WeightsInType&& weights);

  /**
   * Train on the training set and assess performance on the validation set by
   * using the class Metric.
   *
   * @param args Arguments for the given MLAlgorithm taken by its constructor
   *     (in addition to the passed ones in the SimpleCV constructor).
   */
  template<typename... MLAlgorithmArgs>
  double Evaluate(const MLAlgorithmArgs&... args);

  //! Access and modify the last trained model.
  MLAlgorithm& Model();

 private:
  //! A short alias for CVBase.
  using Base = CVBase<MLAlgorithm, MatType, PredictionsType, WeightsType>;

  //! An auxiliary object.
  Base base;

  //! All input data points.
  MatType xs;
  //! All input predictions.
  PredictionsType ys;
  //! All input weights (optional).
  WeightsType weights;

  //! The training data points.
  MatType trainingXs;
  //! The training predictions.
  PredictionsType trainingYs;
  //! The training weights (optional).
  WeightsType trainingWeights;

  //! The validation data points.
  MatType validationXs;
  //! The validation predictions.
  PredictionsType validationYs;

  //! The pointer to the last trained model.
  std::unique_ptr<MLAlgorithm> modelPtr;

  /**
   * Assert data consistency and initialize fields required for running
   * cross-validation.
   */
  template<typename MatInType,
           typename PredictionsInType>
  SimpleCV(Base&& base,
           const double validationSize,
           MatInType&& xs,
           PredictionsInType&& ys);

  /**
   * Assert data consistency and initialize fields required for running
   * cross-validation in the case of weighted learning.
   */
  template<typename MatInType,
           typename PredictionsInType,
           typename WeightsInType>
  SimpleCV(Base&& base,
           const double validationSize,
           MatInType&& xs,
           PredictionsInType&& ys,
           WeightsInType&& weights);

  /**
   * Calculate the number of training points and assert it is legitimate.
   */
  size_t CalculateAndAssertNumberOfTrainingPoints(const double validationSize);

  /**
   * Get the specified submatrix without coping the data.
   */
  template<typename ElementType>
  arma::Mat<ElementType> GetSubset(arma::Mat<ElementType>& m,
                                   const size_t firstCol,
                                   const size_t lastCol);

  /**
   * Get the specified subrow without coping the data.
   */
  template<typename ElementType>
  arma::Row<ElementType> GetSubset(arma::Row<ElementType>& r,
                                   const size_t firstCol,
                                   const size_t lastCol);

  /**
   * Train and run evaluation in the case of non-weighted learning.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = !Base::MIE::SupportsWeights,
           typename = std::enable_if_t<Enabled>>
  double TrainAndEvaluate(const MLAlgorithmArgs&... args);

  /**
   * Train and run evaluation in the case of supporting weighted learning.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = Base::MIE::SupportsWeights,
           typename = std::enable_if_t<Enabled>,
           typename = void>
  double TrainAndEvaluate(const MLAlgorithmArgs&... args);
};

} // namespace mlpack

// Include implementation
#include "simple_cv_impl.hpp"

#endif
