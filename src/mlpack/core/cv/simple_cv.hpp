/**
 * @file simple_cv.hpp
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
namespace cv {

/**
 * The class SimpleCV splits data into training and validation sets, runs
 * training on the training set and evaluates performance on the validation set.
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
class SimpleCV :
    private CVBase<MLAlgorithm, MatType, PredictionsType, WeightsType>
{
public:
  /**
   * This constructor splits data into training and validation sets.
   *
   * @param validationSize A proportion (between 0 and 1) of the data used as a
   *     validation set.
   * @param args Basic constructor arguments for MLAlgortithm (see the CVBase
   *     constructors for reference).
   */
  template<typename... CVBaseArgs>
  SimpleCV(const float validationSize, CVBaseArgs...args);

  /**
   * Train on the training set and assess performance on the validation set by
   * using the class Metric.
   *
   * @param args Arguments for MLAlgorithm (in addition to the passed
   *     ones in the constructor).
   */
  template<typename... MLAlgorithmArgs>
  double Evaluate(const MLAlgorithmArgs& ...args);

  //! Access and modify the trained model.
  MLAlgorithm& Model();

 private:
  using Base = CVBase<MLAlgorithm, MatType, PredictionsType, WeightsType>;

  MatType trainingXs;
  PredictionsType trainingYs;
  WeightsType trainingWeights;

  MatType validationXs;
  PredictionsType validationYs;

  std::unique_ptr<MLAlgorithm> modelPtr;

  /*
   * Initialize without weights.
   */
  template<typename DataArgsTupleT,
           typename = typename std::enable_if<
               std::tuple_size<DataArgsTupleT>::value == 2>::type>
  void Init(const float validationSize, const DataArgsTupleT& dataArgsTuple);

  /*
   * Initialize with weights.
   */
  template<typename DataArgsTupleT,
           typename = typename std::enable_if<
               std::tuple_size<DataArgsTupleT>::value == 3>::type,
           typename = void>
  void Init(const float validationSize, const DataArgsTupleT& dataArgsTuple);

  size_t CalculateAndAssertNumberOfTrainingPoints(const float validationSize,
                                                  const size_t total);

  void InitTrainingAndValidationSets(const MatType& xs,
                                     const PredictionsType& ys,
                                     const size_t numberOfTrainingPoints);

  /*
   * Train and run evaluation in the case of non-weighted learning.
   */
  template<typename...MLAlgorithmArgs,
           bool Enabled = !Base::MIE::SupportsWeights,
           typename = typename std::enable_if<Enabled>::type>
  double TrainAndEvaluate(const MLAlgorithmArgs& ...mlAlgorithmArgs);

  /*
   * Train and run evaluation in the case of supporting weighted learning.
   */
  template<typename...MLAlgorithmArgs,
           bool Enabled = Base::MIE::SupportsWeights,
           typename = typename std::enable_if<Enabled>::type,
           typename = void>
  double TrainAndEvaluate(const MLAlgorithmArgs& ...mlAlgorithmArgs);
};

} // namespace cv
} // namespace mlpack

// Include implementation
#include "simple_cv_impl.hpp"

#endif
