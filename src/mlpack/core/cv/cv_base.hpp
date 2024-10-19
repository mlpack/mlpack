/**
 * @file core/cv/cv_base.hpp
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

/**
 * An auxiliary class for cross-validation. It serves to handle basic non-data
 * constructor parameters of a machine learning algorithm (like datasetInfo or
 * numClasses) and to assert that the machine learning algorithm and data
 * satisfy certain conditions.
 *
 * This class is not meant to be used directly by users. To cross-validate
 * rather use end-user classes like SimpleCV or KFoldCV.
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
  //! A short alias for MetaInfoExtractor.
  using MIE =
      MetaInfoExtractor<MLAlgorithm, MatType, PredictionsType, WeightsType>;

  /**
   * Assert that MLAlgorithm doesn't take any additional basic parameters like
   * numClasses.
   */
  CVBase();

  /**
   * Assert that MLAlgorithm takes the numClasses parameter and store it.
   *
   * @param numClasses Number of classes in the dataset.
   */
  CVBase(const size_t numClasses);

  /**
   * Assert that MLAlgorithm takes the numClasses parameter and a
   * data::DatasetInfo parameter and store them.
   *
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param numClasses Number of classes in the dataset.
   */
  CVBase(const data::DatasetInfo& datasetInfo,
         const size_t numClasses);

  /**
   * Assert there is the equal number of data points and predictions.
   */
  static void AssertDataConsistency(const MatType& xs,
                                    const PredictionsType& ys);

  /**
   * Assert weighted learning is supported and there is the equal number of data
   * points and weights.
   */
  static void AssertWeightsConsistency(const MatType& xs,
                                       const WeightsType& weights);

  /**
   * Train MLAlgorithm with given data points, predictions, and hyperparameters
   * depending on what CVBase constructor has been called.
   */
  template<typename... MLAlgorithmArgs>
  MLAlgorithm Train(const MatType& xs,
                    const PredictionsType& ys,
                    const MLAlgorithmArgs&... args);

  /**
   * Train MLAlgorithm with given data points, predictions, weights, and
   * hyperparameters depending on what CVBase constructor has been called.
   */
  template<typename... MLAlgorithmArgs>
  MLAlgorithm Train(const MatType& xs,
                    const PredictionsType& ys,
                    const WeightsType& weights,
                    const MLAlgorithmArgs&... args);

 private:
  static_assert(MIE::IsSupported,
      "The given MLAlgorithm is not supported by MetaInfoExtractor");

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
   * Construct a trained MLAlgorithm model if MLAlgorithm doesn't take the
   * numClasses parameter.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = !MIE::TakesNumClasses,
           typename = std::enable_if_t<Enabled>>
  MLAlgorithm TrainModel(const MatType& xs,
                         const PredictionsType& ys,
                         const MLAlgorithmArgs&... args);

  /**
   * Construct a trained MLAlgorithm model if MLAlgorithm takes the
   * numClasses parameter.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = MIE::TakesNumClasses & !MIE::TakesDatasetInfo,
           typename = std::enable_if_t<Enabled>,
           typename = void>
  MLAlgorithm TrainModel(const MatType& xs,
                         const PredictionsType& ys,
                         const MLAlgorithmArgs&... args);

  /**
   * Construct a trained MLAlgorithm model if MLAlgorithm takes the
   * numClasses parameter and a data::DatasetInfo parameter.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = MIE::TakesNumClasses & MIE::TakesDatasetInfo,
           typename = std::enable_if_t<Enabled>,
           typename = void,
           typename = void>
  MLAlgorithm TrainModel(const MatType& xs,
                         const PredictionsType& ys,
                         const MLAlgorithmArgs&... args);

  /**
   * Construct a trained MLAlgorithm model if MLAlgorithm doesn't take the
   * numClasses parameter.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = !MIE::TakesNumClasses,
           typename = std::enable_if_t<Enabled>>
  MLAlgorithm TrainModel(const MatType& xs,
                         const PredictionsType& ys,
                         const WeightsType& weights,
                         const MLAlgorithmArgs&... args);

  /**
   * Construct a trained MLAlgorithm model if MLAlgorithm takes the
   * numClasses parameter.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = MIE::TakesNumClasses & !MIE::TakesDatasetInfo,
           typename = std::enable_if_t<Enabled>,
           typename = void>
  MLAlgorithm TrainModel(const MatType& xs,
                         const PredictionsType& ys,
                         const WeightsType& weights,
                         const MLAlgorithmArgs&... args);

  /**
   * Construct a trained MLAlgorithm model if MLAlgorithm takes the
   * numClasses parameter and a data::DatasetInfo parameter.
   */
  template<typename... MLAlgorithmArgs,
           bool Enabled = MIE::TakesNumClasses & MIE::TakesDatasetInfo,
           typename = std::enable_if_t<Enabled>,
           typename = void,
           typename = void>
  MLAlgorithm TrainModel(const MatType& xs,
                         const PredictionsType& ys,
                         const WeightsType& weights,
                         const MLAlgorithmArgs&... args);

  /**
   * When MLAlgorithm supports a data::DatasetInfo parameter, training should be
   * treated separately - there are models that can be constructed with and
   * without a data:DatasetInfo parameter and models that can be constructed
   * only with a data::DatasetInfo parameter.
   *
   * Construct a trained MLAlgorithm model when it can be constructed without a
   * data::DatasetInfo parameter.
   */
  template<bool ConstructableWithoutDatasetInfo,
           typename... MLAlgorithmArgs,
           typename = std::enable_if_t<ConstructableWithoutDatasetInfo>>
  MLAlgorithm TrainModel(const MatType& xs,
                         const PredictionsType& ys,
                         const MLAlgorithmArgs&... args);

  /**
   * Construct a trained MLAlgorithm model when it can't be constructed without
   * a data::DatasetInfo parameter.
   */
  template<bool ConstructableWithoutDatasetInfo,
           typename... MLAlgorithmArgs,
           typename = std::enable_if_t<!ConstructableWithoutDatasetInfo>,
           typename = void>
  MLAlgorithm TrainModel(const MatType& xs,
                         const PredictionsType& ys,
                         const MLAlgorithmArgs&... args);
};

} // namespace mlpack

// Include implementation
#include "cv_base_impl.hpp"

#endif
