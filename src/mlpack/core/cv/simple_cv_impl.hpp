/**
 * @file core/cv/simple_cv_impl.hpp
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

namespace mlpack {

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MIT, typename PIT>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(const double validationSize,
                                MIT&& xs,
                                PIT&& ys) :
    SimpleCV(Base(), validationSize, std::forward<MIT>(xs),
        std::forward<PIT>(ys))
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MIT, typename PIT>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(const double validationSize,
                                MIT&& xs,
                                PIT&& ys,
                                const size_t numClasses) :
    SimpleCV(Base(numClasses), validationSize, std::forward<MIT>(xs),
        std::forward<PIT>(ys))
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MIT, typename PIT>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(const double validationSize,
                                MIT&& xs,
                                const data::DatasetInfo& datasetInfo,
                                PIT&& ys,
                                const size_t numClasses) :
    SimpleCV(Base(datasetInfo, numClasses), validationSize,
        std::forward<MIT>(xs), std::forward<PIT>(ys))
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MIT, typename PIT, typename WIT>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(const double validationSize,
                                MIT&& xs,
                                PIT&& ys,
                                WIT&& weights) :
    SimpleCV(Base(), validationSize, std::forward<MIT>(xs),
        std::forward<PIT>(ys), std::forward<WIT>(weights))
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MIT, typename PIT, typename WIT>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(const double validationSize,
                                MIT&& xs,
                                PIT&& ys,
                                const size_t numClasses,
                                WIT&& weights) :
    SimpleCV(Base(numClasses), validationSize, std::forward<MIT>(xs),
        std::forward<PIT>(ys), std::forward<WIT>(weights))
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MIT, typename PIT, typename WIT>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(const double validationSize,
                                MIT&& xs,
                                const data::DatasetInfo& datasetInfo,
                                PIT&& ys,
                                const size_t numClasses,
                                WIT&& weights) :
    SimpleCV(Base(datasetInfo, numClasses), validationSize,
        std::forward<MIT>(xs), std::forward<PIT>(ys),
        std::forward<WIT>(weights))
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MIT, typename PIT>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(Base&& base,
                                const double validationSize,
                                MIT&& xs,
                                PIT&& ys) :
    base(std::move(base)),
    xs(std::forward<MIT>(xs)),
    ys(std::forward<PIT>(ys))
{
  Base::AssertDataConsistency(this->xs, this->ys);

  size_t numberOfTrainingPoints = CalculateAndAssertNumberOfTrainingPoints(
      validationSize);

  trainingXs = GetSubset(this->xs, 0, numberOfTrainingPoints - 1);
  trainingYs = GetSubset(this->ys, 0, numberOfTrainingPoints - 1);

  validationXs = GetSubset(this->xs, numberOfTrainingPoints, xs.n_cols - 1);
  validationYs = GetSubset(this->ys, numberOfTrainingPoints, xs.n_cols - 1);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MIT, typename PIT, typename WIT>
SimpleCV<MLAlgorithm,
         Metric,
         MatType,
         PredictionsType,
         WeightsType>::SimpleCV(Base&& base,
                                const double validationSize,
                                MIT&& xs,
                                PIT&& ys,
                                WIT&& weights) :
    SimpleCV(std::move(base), validationSize, std::forward<MIT>(xs),
        std::forward<PIT>(ys))
{
  this->weights = std::forward<WIT>(weights);

  Base::AssertWeightsConsistency(this->xs, this->weights);

  trainingWeights = GetSubset(this->weights, 0, trainingXs.n_cols - 1);
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
template<typename ElementType>
arma::Mat<ElementType> SimpleCV<MLAlgorithm,
                                Metric,
                                MatType,
                                PredictionsType,
                                WeightsType>::GetSubset(
    arma::Mat<ElementType>& m,
    const size_t firstCol,
    const size_t lastCol)
{
  return arma::Mat<ElementType>(m.colptr(firstCol), m.n_rows,
      lastCol - firstCol + 1, false, true);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename ElementType>
arma::Row<ElementType> SimpleCV<MLAlgorithm,
                                Metric,
                                MatType,
                                PredictionsType,
                                WeightsType>::GetSubset(
    arma::Row<ElementType>& r,
    const size_t firstCol,
    const size_t lastCol)
{
  return arma::Row<ElementType>(r.colptr(firstCol), lastCol - firstCol + 1,
      false, true);
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
  modelPtr.reset(new MLAlgorithm(base.Train(trainingXs, trainingYs, args...)));

  return Metric::Evaluate(*modelPtr, validationXs, validationYs);
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
    modelPtr.reset(new MLAlgorithm(
        base.Train(trainingXs, trainingYs, trainingWeights, args...)));
  else
    modelPtr.reset(new MLAlgorithm(
        base.Train(trainingXs, trainingYs, args...)));

  return Metric::Evaluate(*modelPtr, validationXs, validationYs);
}

} // namespace mlpack

#endif
