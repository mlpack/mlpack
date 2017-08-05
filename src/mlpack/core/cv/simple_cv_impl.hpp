/**
 * @file simple_cv_impl.hpp
 * @author Kirill Mishchenko
 *
 * The implementation of the class SimpleCV.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_SIMPLE_CV_IMPL_HPP
#define MLPACK_CORE_CV_SIMPLE_CV_IMPL_HPP

#include <cmath>

namespace mlpack {
namespace cv {

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MatInType, typename PredictionsInType>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(const double validationSize,
                                const MatInType& xs,
                                const PredictionsInType& ys) :
    xs(xs),
    ys(ys)
{
  Base::AssertDataConsistency(this->xs, this->ys);
  InitTrainingAndValidationSets(validationSize);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MatInType, typename PredictionsInType>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(const double validationSize,
                                const MatInType& xs,
                                const PredictionsInType& ys,
                                const size_t numClasses) :
    base(numClasses),
    xs(xs),
    ys(ys)
{
  Base::AssertDataConsistency(this->xs, this->ys);
  InitTrainingAndValidationSets(validationSize);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MatInType, typename PredictionsInType>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(const double validationSize,
                                const MatInType& xs,
                                const data::DatasetInfo& datasetInfo,
                                const PredictionsInType& ys,
                                const size_t numClasses) :
    base(datasetInfo, numClasses),
    xs(xs),
    ys(ys)
{
  Base::AssertDataConsistency(this->xs, this->ys);
  InitTrainingAndValidationSets(validationSize);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MatInType, typename PredictionsInType, typename WeightsInType>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(const double validationSize,
                                const MatInType& xs,
                                const PredictionsInType& ys,
                                const WeightsInType& weights) :
    xs(xs),
    ys(ys),
    weights(weights)
{
  Base::AssertDataConsistency(this->xs, this->ys, this->weights);
  InitTrainingAndValidationSetsWithWeights(validationSize);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MatInType, typename PredictionsInType, typename WeightsInType>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(const double validationSize,
                                const MatInType& xs,
                                const PredictionsInType& ys,
                                const size_t numClasses,
                                const WeightsInType& weights) :
    base(numClasses),
    xs(xs),
    ys(ys),
    weights(weights)
{
  Base::AssertDataConsistency(this->xs, this->ys, this->weights);
  InitTrainingAndValidationSetsWithWeights(validationSize);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MatInType, typename PredictionsInType, typename WeightsInType>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(const double validationSize,
                                const MatInType& xs,
                                const data::DatasetInfo& datasetInfo,
                                const PredictionsInType& ys,
                                const size_t numClasses,
                                const WeightsInType& weights) :
    base(datasetInfo, numClasses),
    xs(xs),
    ys(ys),
    weights(weights)
{
  Base::AssertDataConsistency(this->xs, this->ys, this->weights);
  InitTrainingAndValidationSetsWithWeights(validationSize);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs>
double SimpleCV<MLAlgorithm,
                Metric,
                MatType,
                PredictionsType,
                WeightsType>::Evaluate(const MLAlgorithmArgs&... args)
{
  return TrainAndEvaluate(args...);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
void SimpleCV<MLAlgorithm,
              Metric,
              MatType,
              PredictionsType,
              WeightsType>::InitTrainingAndValidationSets(
    const double validationSize)
{
  size_t numberOfTrainingPoints = CalculateAndAssertNumberOfTrainingPoints(
      validationSize);

  trainingXs = GetSubset(xs, 0, numberOfTrainingPoints - 1);
  trainingYs = GetSubset(ys, 0, numberOfTrainingPoints - 1);

  validationXs = GetSubset(xs, numberOfTrainingPoints, xs.n_cols - 1);
  validationYs = GetSubset(ys, numberOfTrainingPoints, xs.n_cols - 1);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
void SimpleCV<MLAlgorithm,
              Metric,
              MatType,
              PredictionsType,
              WeightsType>::InitTrainingAndValidationSetsWithWeights(
    const double validationSize)
{
  InitTrainingAndValidationSets(validationSize);
  trainingWeights = GetSubset(weights, 0, trainingXs.n_cols - 1);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
size_t SimpleCV<MLAlgorithm,
                Metric,
                MatType,
                PredictionsType,
                WeightsType>::CalculateAndAssertNumberOfTrainingPoints(
    const double validationSize)
{
  if (validationSize < 0.0 || validationSize > 1.0)
    throw std::invalid_argument("SimpleCV: the validationSize parameter should "
        "be more than 0 and less than 1");

  if (xs.n_cols < 2)
    throw std::invalid_argument("SimpleCV: 2 or more data points are expected");

  size_t trainingPoints = round(xs.n_cols * (1.0 - validationSize));

  if (trainingPoints == 0 || trainingPoints == xs.n_cols)
    throw std::invalid_argument("SimpleCV: the validationSize parameter is "
        "either too small or too big");

  return trainingPoints;
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename>
double SimpleCV<MLAlgorithm,
                Metric,
                MatType,
                PredictionsType,
                WeightsType>::TrainAndEvaluate(const MLAlgorithmArgs&... args)
{
  base.Train(model, trainingXs, trainingYs, args...);

  return Metric::Evaluate(model, validationXs, validationYs);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename, typename>
double SimpleCV<MLAlgorithm,
                Metric,
                MatType,
                PredictionsType,
                WeightsType>::TrainAndEvaluate(const MLAlgorithmArgs&... args)
{
  if (trainingWeights.n_elem > 0)
    base.Train(model, trainingXs, trainingYs, trainingWeights, args...);
  else
    base.Train(model, trainingXs, trainingYs, args...);

  return Metric::Evaluate(model, validationXs, validationYs);
}

} // namespace cv
} // namespace mlpack

#endif
