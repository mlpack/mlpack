/**
 * @file
 * The implementation of k-fold cross-validation base class.
 * 
 * @authors Kirill Mishchenko, Felix Patschkowski
 *
 * @copyright mlpack is free software; you may redistribute it and/or modify it
 * under the terms of the 3-clause BSD license.  You should have received a copy
 * of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_K_FOLD_CV_BASE_IMPL_HPP
#define MLPACK_CORE_CV_K_FOLD_CV_BASE_IMPL_HPP

namespace mlpack {

template<
  typename Derived,
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::KFoldCVBase(const size_t k,
    const MatType& xs,
    const PredictionsType& ys) :
  KFoldCVBase(Base(), k, xs, ys)
{ /* Nothing left to do. */
}

template<
  typename Derived, 
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::KFoldCVBase(const size_t k,
    const MatType& xs,
    const PredictionsType& ys,
    const size_t numClasses) :
  KFoldCVBase(Base(numClasses), k, xs, ys)
{ /* Nothing left to do. */
}

template<
  typename Derived, 
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::KFoldCVBase(const size_t k,
    const MatType& xs,
    const data::DatasetInfo& datasetInfo,
    const PredictionsType& ys,
    const size_t numClasses) :
  KFoldCVBase(Base(datasetInfo, numClasses), k, xs, ys)
{ /* Nothing left to do. */
}

template<
  typename Derived, 
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::KFoldCVBase(const size_t k,
    const MatType& xs,
    const PredictionsType& ys,
    const WeightsType& weights) :
  KFoldCVBase(Base(), k, xs, ys, weights)
{ /* Nothing left to do. */
}

template<
  typename Derived, 
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::KFoldCVBase(const size_t k,
    const MatType& xs,
    const PredictionsType& ys,
    const size_t numClasses,
    const WeightsType& weights) :
  KFoldCVBase(Base(numClasses), k, xs, ys, weights)
{ /* Nothing left to do. */
}

template<
  typename Derived, 
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::KFoldCVBase(const size_t k,
    const MatType& xs,
    const data::DatasetInfo& datasetInfo,
    const PredictionsType& ys,
    const size_t numClasses,
    const WeightsType& weights) :
  KFoldCVBase(Base(datasetInfo, numClasses), k, xs, ys, weights)
{ /* Nothing left to do. */
}

template<
  typename Derived,
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::KFoldCVBase(Base&& base,
    const size_t k,
    const MatType& xs,
    const PredictionsType& ys) :
  base(std::move(base)),
  k(k)
{
  if (k < 2)
    throw std::invalid_argument("KFoldCVBase: k should not be less than 2");

  Base::AssertDataConsistency(xs, ys);

  InitKFoldCVMat(xs, this->xs);
  InitKFoldCVMat(ys, this->ys);
}

template<
  typename Derived,
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::KFoldCVBase(Base&& base,
    const size_t k,
    const MatType& xs,
    const PredictionsType& ys,
    const WeightsType& weights) :
  base(std::move(base)),
  k(k)
{
  Base::AssertWeightsConsistency(xs, weights);

  InitKFoldCVMat(xs, this->xs);
  InitKFoldCVMat(ys, this->ys);
  InitKFoldCVMat(weights, this->weights);
}

template<
  typename Derived, 
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
template<typename... MLAlgorithmArgs>
double KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::Evaluate(const MLAlgorithmArgs&... args)
{
  return TrainAndEvaluate(args...);
}

template<
  typename Derived, 
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
MLAlgorithm& KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::Model()
{
  if (modelPtr == nullptr)
    throw std::logic_error(
      "KFoldCVBase::Model(): attempted to access an uninitialized model");

  return *modelPtr;
}

template<
  typename Derived,
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
template<typename DataType>
void KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::InitKFoldCVMat(const DataType& source,
    DataType& destination)
{
  binSize = source.n_cols / k;
  lastBinSize = source.n_cols - ((k - 1) * binSize);

  destination = (k == 2) ? source : join_rows(source,
    source.cols(0, source.n_cols - lastBinSize - 1));
}

template<
  typename Derived, 
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename>
double KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::TrainAndEvaluate(const MLAlgorithmArgs&... args)
{
  arma::vec evaluations(k);

  size_t numInvalidScores = 0;
  for (size_t i = 0; i < k; ++i)
  {
    MLAlgorithm&& model = base.Train(
      static_cast<Derived*>(this)->GetTrainingSubset(xs, i),
      static_cast<Derived*>(this)->GetTrainingSubset(ys, i), args...);
    evaluations(i) = Metric::Evaluate(model,
      static_cast<Derived*>(this)->GetValidationSubset(xs, i),
      static_cast<Derived*>(this)->GetValidationSubset(ys, i));
    if (std::isnan(evaluations(i)) || std::isinf(evaluations(i)))
    {
      ++numInvalidScores;
      Log::Warn << "KFoldCVBase::TrainAndEvaluate(): fold " << i << " returned "
        << "a score of " << evaluations(i) << "; ignoring when computing "
        << "the average score." << std::endl;
    }
    if (i == k - 1)
      modelPtr.reset(new MLAlgorithm(std::move(model)));
  }

  if (numInvalidScores == k)
  {
    Log::Warn << "KFoldCVBase::TrainAndEvaluate(): all folds returned invalid "
      << "scores!  Returning 0.0 as overall score." << std::endl;
    return 0.0;
  }

  return arma::mean(evaluations.elem(arma::find_finite(evaluations)));
}

template<
  typename Derived,
  typename MLAlgorithm,
  typename Metric,
  typename MatType,
  typename PredictionsType,
  typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename, typename>
double KFoldCVBase<Derived,
  MLAlgorithm,
  Metric,
  MatType,
  PredictionsType,
  WeightsType>::TrainAndEvaluate(const MLAlgorithmArgs&... args)
{
  arma::vec evaluations(k);

  for (size_t i = 0; i < k; ++i)
  {
    MLAlgorithm&& model = (weights.n_elem > 0) ?
      base.Train(static_cast<Derived*>(this)->GetTrainingSubset(xs, i),
        static_cast<Derived*>(this)->GetTrainingSubset(ys, i),
        static_cast<Derived*>(this)->GetTrainingSubset(weights, i), args...) :
      base.Train(static_cast<Derived*>(this)->GetTrainingSubset(xs, i),
        static_cast<Derived*>(this)->GetTrainingSubset(ys, i),
        args...);
    evaluations(i) = Metric::Evaluate(model, 
      static_cast<Derived*>(this)->GetValidationSubset(xs, i),
      static_cast<Derived*>(this)->GetValidationSubset(ys, i));
    if (i == k - 1)
      modelPtr.reset(new MLAlgorithm(std::move(model)));
  }

  return arma::mean(evaluations);
}

} // namespace mlpack

#endif
