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
template<typename... CVBaseArgs>
SimpleCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>::SimpleCV(const float validationSize, CVBaseArgs... args) :
    Base(args...)
{
  Init(validationSize, Base::ExtractDataArgs(args...));
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
MLAlgorithm& SimpleCV<MLAlgorithm,
                      Metric,
                      MatType,
                      PredictionsType,
                      WeightsType>::Model()
{
  if (modelPtr == nullptr)
    throw std::logic_error(
        "SimpleCV::Model(): attempted to access an uninitialized model");

  return *modelPtr;
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename DataArgsTupleT,
         typename>
void SimpleCV<MLAlgorithm,
              Metric,
              MatType,
              PredictionsType,
              WeightsType>::Init(const float validationSize,
                                 const DataArgsTupleT& dataArgsTuple)
{
  const MatType& xs = std::get<0>(dataArgsTuple);
  const PredictionsType& ys = std::get<1>(dataArgsTuple);

  Base::AssertDataConsistency(xs, ys);

  size_t numberOfTrainingPoints = CalculateAndAssertNumberOfTrainingPoints(
      validationSize, xs.n_cols);

  InitTrainingAndValidationSets(xs, ys, numberOfTrainingPoints);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename DataArgsTupleT,
         typename,
         typename>
void SimpleCV<MLAlgorithm,
              Metric,
              MatType,
              PredictionsType,
              WeightsType>::Init(const float validationSize,
                                 const DataArgsTupleT& dataArgsTuple)
{
  const MatType& xs = std::get<0>(dataArgsTuple);
  const PredictionsType& ys = std::get<1>(dataArgsTuple);
  const WeightsType& weights = std::get<2>(dataArgsTuple);

  Base::AssertDataConsistency(xs, ys, weights);

  size_t numberOfTrainingPoints = CalculateAndAssertNumberOfTrainingPoints(
      validationSize, xs.n_cols);

  InitTrainingAndValidationSets(xs, ys, numberOfTrainingPoints);

  trainingWeights = weights.cols(0, numberOfTrainingPoints - 1);
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
    const float validationSize,
    const size_t total)
{
  if (validationSize < 0.0F || validationSize > 1.0F)
  {
    std::ostringstream oss;
    oss << "SimpleCV: the validationSize parameter should be "
        << "more than 0 and less than 1" << std::endl;
    throw std::invalid_argument(oss.str());
  }

  if (total < 2)
  {
    std::ostringstream oss;
    oss << "SimpleCV: 2 or more data points are expected" << std::endl;
    throw std::invalid_argument(oss.str());
  }

  size_t trainingPoints = roundf(total * (1.0F - validationSize));

  if (trainingPoints == 0 || trainingPoints == total)
  {
    std::ostringstream oss;
    oss << "SimpleCV: the validationSize parameter is either too small "
        << "or too big" << std::endl;
    throw std::invalid_argument(oss.str());
  }

  return trainingPoints;
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool, typename>
double SimpleCV<MLAlgorithm,
                Metric,
                MatType,
                PredictionsType,
                WeightsType>::TrainAndEvaluate(const MLAlgorithmArgs&... args)
{
  modelPtr = this->Train(trainingXs, trainingYs, args...);

  return Metric::Evaluate(*modelPtr, validationXs, validationYs);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool, typename, typename>
double SimpleCV<MLAlgorithm,
                Metric,
                MatType,
                PredictionsType,
                WeightsType>::TrainAndEvaluate(const MLAlgorithmArgs&... args)
{
  if (trainingWeights.n_elem > 0)
    modelPtr = this->Train(trainingXs, trainingYs, trainingWeights, args...);
  else
    modelPtr = this->Train(trainingXs, trainingYs, args...);

  return Metric::Evaluate(*modelPtr, validationXs, validationYs);
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
    const MatType& xs,
    const PredictionsType& ys,
    const size_t numberOfTrainingPoints)
{
  trainingXs = xs.cols(0, numberOfTrainingPoints - 1);
  trainingYs = ys.cols(0, numberOfTrainingPoints - 1);

  validationXs = xs.cols(numberOfTrainingPoints, xs.n_cols - 1);
  validationYs = ys.cols(numberOfTrainingPoints, xs.n_cols - 1);
}

} // namespace cv
} // namespace mlpack

#endif
