/**
 * @file cv_base.hpp
 * @author Kirill Mishchenko
 *
 * A base class for cross-validation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_CV_BASE_HPP
#define MLPACK_CORE_CV_CV_BASE_HPP

#include <mlpack/core/cv/meta_info_extractor.hpp>

namespace mlpack {
namespace cv {

/**
 * A base class for cross-validation. It serves to handle basic non-data
 * constructor parameters of a machine learning algorithm (like datasetInfo or
 * numClasses) and to assert that the machine learning algorithm and data
 * satisfy certain conditions.
 *
 * This class is not meant to be used directly by users. Rather use the CVBase
 * constructors as a reference for what additional arguments are accepted by
 * cross-validation strategies like SimpleCV or KFoldCV.
 *
 * @tparam MLAlgorithm A machine learning algorithm.
 * @tparam MatType The type of data.
 * @tparam PredictionsType The type of predictions (labels/responses).
 * @tparam WeightsType The type of weights. It supposed to be void* when weights
 *     are not supported.
 */
template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
class CVBase
{
 public:
  /**
   * This constructor can be used for regression algorithms and for binary
   * classification algorithms.
   *
   * @param xs Dataset to cross-validate on.
   * @param ys Predictions (labels for classification algorithms and responses
   *     for regression algorithms) for each point from the dataset.
   *
   * @tparam MatInType A type that can be converted to MatType.
   * @tparam PredictionsInType A type that can be converted to PredictionsType.
   */
  template<typename MatInType, typename PredictionsInType>
  CVBase(const MatInType& xs,
         const PredictionsInType& ys);

  /**
   * This constructor can be used for multiclass classification algorithms.
   *
   * @param xs Dataset to cross-validate on.
   * @param ys Labels for each point from the dataset.
   * @param numClasses Number of classes in the dataset.
   *
   * @tparam MatInType A type that can be converted to MatType.
   * @tparam PredictionsInType A type that can be converted to PredictionsType.
   */
  template<typename MatInType, typename PredictionsInType>
  CVBase(const MatInType& xs,
         const PredictionsInType& ys,
         const size_t numClasses);

  /**
   * This constructor can be used for multiclass classification algorithms that
   * can take a data::DatasetInfo parameter.
   *
   * @param xs Dataset to cross-validate on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param ys Labels for each point from the dataset.
   * @param numClasses Number of classes in the dataset.
   *
   * @tparam MatInType A type that can be converted to MatType.
   * @tparam PredictionsInType A type that can be converted to PredictionsType.
   */
  template<typename MatInType, typename PredictionsInType>
  CVBase(const MatInType& xs,
         const data::DatasetInfo& datasetInfo,
         const PredictionsInType& ys,
         const size_t numClasses);

  /**
   * This constructor can be used for regression and binary classification
   * algorithms that support weighted learning.
   *
   * @param xs Dataset to cross-validate on.
   * @param ys Predictions (labels for classification algorithms and responses
   *     for regression algorithms) for each point from the dataset.
   * @param weights Observation weights (for boosting).
   *
   * @tparam MatInType A type that can be converted to MatType.
   * @tparam PredictionsInType A type that can be converted to PredictionsType.
   * @tparam WeightsInType A type that can be converted to WeightsType.
   */
  template<typename MatInType,
           typename PredictionsInType,
           typename WeightsInType>
  CVBase(const MatInType& xs,
         const PredictionsInType& ys,
         const WeightsInType& weights);

  /**
   * This constructor can be used for multiclass classification algorithms that
   * support weighted learning.
   *
   * @param xs Dataset to cross-validate on.
   * @param ys Labels for each point from the dataset.
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
  CVBase(const MatInType& xs,
         const PredictionsInType& ys,
         const size_t numClasses,
         const WeightsInType& weights);

  /**
   * This constructor can be used for multiclass classification algorithms that
   * can take a data::DatasetInfo parameter and support weighted learning.
   *
   * @param xs Dataset to cross-validate on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param ys Labels for each point from the dataset.
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
  CVBase(const MatInType& xs,
         const data::DatasetInfo& datasetInfo,
         const PredictionsInType& ys,
         const size_t numClasses,
         const WeightsInType& weights);

 protected:
  using MIE =
      MetaInfoExtractor<MLAlgorithm, MatType, PredictionsType, WeightsType>;

  static_assert(MIE::IsSupported,
      "The given MLAlgorithm is not supported by MetaInfoExtractor");

  /**
   * A set of methods for extracting input data arguments. It is supposed to be
   * called with variadic template arguments like ExtractDataArgs(args...).
   */
  template<typename MatInType, typename PredictionsInType>
  static std::tuple<const MatInType&, const PredictionsInType&> ExtractDataArgs(
      const MatInType& xs,
      const PredictionsInType& ys)
  { return std::tuple<const MatInType&, const PredictionsInType&>(xs, ys); }

  template<typename MatInType, typename PredictionsInType>
  static std::tuple<const MatInType&, const PredictionsInType&> ExtractDataArgs(
      const MatInType& xs,
      const PredictionsInType& ys,
      const size_t /* numClasses */)
  { return std::tuple<const MatInType&, const PredictionsInType&>(xs, ys); }

  template<typename MatInType, typename PredictionsInType>
  static std::tuple<const MatInType&, const PredictionsInType&> ExtractDataArgs(
      const MatInType& xs,
      const data::DatasetInfo& /* datasetInfo */,
      const PredictionsInType& ys,
      const size_t /* numClasses */)
  { return std::tuple<const MatInType&, const PredictionsInType&>(xs, ys); }

  template<typename MatInType,
           typename PredictionsInType,
           typename WeightsInType>
  static std::tuple<const MatInType&,
                    const PredictionsInType&,
                    const WeightsInType&> ExtractDataArgs(
      const MatInType& xs,
      const PredictionsInType& ys,
      const WeightsInType& weights)
  {
    return std::tuple<const MatInType&, const PredictionsInType&,
        const WeightsInType&>(xs, ys, weights);
  }

  template<typename MatInType,
           typename PredictionsInType,
           typename WeightsInType>
  static std::tuple<const MatInType&,
                    const PredictionsInType&,
                    const WeightsInType&> ExtractDataArgs(
      const MatInType& xs,
      const PredictionsInType& ys,
      const size_t /* numClasses */,
      const WeightsInType& weights)
  {
    return std::tuple<const MatInType&, const PredictionsInType&,
        const WeightsInType&>(xs, ys, weights);
  }

  template<typename MatInType,
           typename PredictionsInType,
           typename WeightsInType>
  static std::tuple<const MatInType&,
                    const PredictionsInType&,
                    const WeightsInType&> ExtractDataArgs(
      const MatInType& xs,
      const data::DatasetInfo& /* datasetInfo */,
      const PredictionsInType& ys,
      const size_t /* numClasses */,
      const WeightsInType& weights)
  {
    return std::tuple<const MatInType&, const PredictionsInType&,
        const WeightsInType&>(xs, ys, weights);
  }

  /**
   * Assert there is an equal number of data points and predictions.
   */
  static void AssertDataConsistency(const MatType& xs,
                                    const PredictionsType& ys);

  /**
   * Assert there is an equal number of data points, predictions, and weights.
   */
  static void AssertDataConsistency(const MatType& xs,
                                    const PredictionsType& ys,
                                    const WeightsType& weights);

  /**
   * Train MLAlgorithm with given data points, predictions, and hyperparameters
   * depending on what CVBase constructor has been called.
   */
  template<typename... MLAlgorithmArgs>
  std::unique_ptr<MLAlgorithm> Train(const MatType& xs,
                                     const PredictionsType& ys,
                                     const MLAlgorithmArgs&... args);

  /**
   * Train MLAlgorithm with given data points, predictions, weights, and
   * hyperparameters depending on what CVBase constructor has been called.
   */
  template<typename... MLAlgorithmArgs>
  std::unique_ptr<MLAlgorithm> Train(const MatType& xs,
                                     const PredictionsType& ys,
                                     const WeightsType& weights,
                                     const MLAlgorithmArgs&... args);

 private:
  //! A variable for storing a data::DatasetInfo parameter if it is passed.
  const data::DatasetInfo datasetInfo;
  //! An indicator whether a data::DatasetInfo parameter has been passed.
  const bool isDatasetInfoPassed;
  //! A variable for storing the numClasses parameter if it is passed.
  size_t numClasses;

  /**
   * Assert there is an equal number of data points and predictions.
   */
  static void AssertSizeEquality(const MatType& xs,
                                 const PredictionsType& ys);

  /**
   * Assert the number of weights is the same as the number of data points.
   */
  static void AssertWeightsSize(const MatType& xs,
                                const WeightsType& weights);

  /**
   * A set of methods for training models depending on what parameters are
   * optional and what parameters are required for MLAlgorithm.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = !MIE::TakesNumClasses,
           typename = typename std::enable_if<Enabled>::type>
  std::unique_ptr<MLAlgorithm> TrainModel(const MatType& xs,
                                          const PredictionsType& ys,
                                          const MLAlgorithmArgs&... args);

  template<typename... MLAlgorithmArgs,
           bool Enabled = MIE::TakesNumClasses & !MIE::TakesDatasetInfo,
           typename = typename std::enable_if<Enabled>::type,
           typename = void>
  std::unique_ptr<MLAlgorithm> TrainModel(const MatType& xs,
                                          const PredictionsType& ys,
                                          const MLAlgorithmArgs&... args);

  template<typename... MLAlgorithmArgs,
           bool Enabled = MIE::TakesNumClasses & MIE::TakesDatasetInfo,
           typename = typename std::enable_if<Enabled>::type,
           typename = void,
           typename = void>
  std::unique_ptr<MLAlgorithm> TrainModel(const MatType& xs,
                                          const PredictionsType& ys,
                                          const MLAlgorithmArgs&... args);

  template<typename... MLAlgorithmArgs,
           bool Enabled = !MIE::TakesNumClasses,
           typename = typename std::enable_if<Enabled>::type>
  std::unique_ptr<MLAlgorithm> TrainModel(const MatType& xs,
                                          const PredictionsType& ys,
                                          const WeightsType& weights,
                                          const MLAlgorithmArgs&... args);

  template<typename... MLAlgorithmArgs,
           bool Enabled = MIE::TakesNumClasses & !MIE::TakesDatasetInfo,
           typename = typename std::enable_if<Enabled>::type,
           typename = void>
  std::unique_ptr<MLAlgorithm> TrainModel(const MatType& xs,
                                          const PredictionsType& ys,
                                          const WeightsType& weights,
                                          const MLAlgorithmArgs&... args);

  template<typename... MLAlgorithmArgs,
           bool Enabled = MIE::TakesNumClasses & MIE::TakesDatasetInfo,
           typename = typename std::enable_if<Enabled>::type,
           typename = void,
           typename = void>
  std::unique_ptr<MLAlgorithm> TrainModel(const MatType& xs,
                                          const PredictionsType& ys,
                                          const WeightsType& weights,
                                          const MLAlgorithmArgs&... args);

  /**
   * When MLAlgorithm supports a data::DatasetInfo parameter, training should be
   * treated separately - there are models that can be constructed with and
   * without a data:DatasetInfo parameter and models that can be constructed
   * only with a data::DatasetInfo parameter.
   */
  template<bool ConstructableWithoutDatasetInfo,
           typename... MLAlgorithmArgs,
           typename =
               typename std::enable_if<ConstructableWithoutDatasetInfo>::type>
  std::unique_ptr<MLAlgorithm> TrainModel(const MatType& xs,
                                          const PredictionsType& ys,
                                          const MLAlgorithmArgs&... args);

  template<bool ConstructableWithoutDatasetInfo,
           typename... MLAlgorithmArgs,
           typename =
               typename std::enable_if<!ConstructableWithoutDatasetInfo>::type,
           typename = void>
  std::unique_ptr<MLAlgorithm> TrainModel(const MatType& xs,
                                          const PredictionsType& ys,
                                          const MLAlgorithmArgs&... args);
};

} // namespace cv
} // namespace mlpack

// Include implementation
#include "cv_base_impl.hpp"

#endif
