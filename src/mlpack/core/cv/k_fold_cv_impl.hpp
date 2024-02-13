/**
 * @file core/cv/k_fold_cv_impl.hpp
 * @author Kirill Mishchenko
 *
 * The implementation of k-fold cross-validation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_K_FOLD_CV_IMPL_HPP
#define MLPACK_CORE_CV_K_FOLD_CV_IMPL_HPP

namespace mlpack {

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>::KFoldCV(const size_t k,
                              const MatType& xs,
                              const PredictionsType& ys,
                              const bool shuffle) :
    KFoldCV(Base(), k, xs, ys, shuffle)
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>::KFoldCV(const size_t k,
                              const MatType& xs,
                              const PredictionsType& ys,
                              const size_t numClasses,
                              const bool shuffle) :
    KFoldCV(Base(numClasses), k, xs, ys, shuffle)
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>::KFoldCV(const size_t k,
                              const MatType& xs,
                              const data::DatasetInfo& datasetInfo,
                              const PredictionsType& ys,
                              const size_t numClasses,
                              const bool shuffle) :
    KFoldCV(Base(datasetInfo, numClasses), k, xs, ys, shuffle)
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>::KFoldCV(const size_t k,
                              const MatType& xs,
                              const PredictionsType& ys,
                              const WeightsType& weights,
                              const bool shuffle) :
    KFoldCV(Base(), k, xs, ys, weights, shuffle)
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>::KFoldCV(const size_t k,
                              const MatType& xs,
                              const PredictionsType& ys,
                              const size_t numClasses,
                              const WeightsType& weights,
                              const bool shuffle) :
    KFoldCV(Base(numClasses), k, xs, ys, weights, shuffle)
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>::KFoldCV(const size_t k,
                              const MatType& xs,
                              const data::DatasetInfo& datasetInfo,
                              const PredictionsType& ys,
                              const size_t numClasses,
                              const WeightsType& weights,
                              const bool shuffle) :
    KFoldCV(Base(datasetInfo, numClasses), k, xs, ys, weights, shuffle)
{ /* Nothing left to do. */ }

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>::KFoldCV(Base&& base,
                              const size_t k,
                              const MatType& xs,
                              const PredictionsType& ys,
                              const bool shuffle) :
    base(std::move(base)),
    k(k)
{
  if (k < 2)
    throw std::invalid_argument("KFoldCV: k should not be less than 2");

  Base::AssertDataConsistency(xs, ys);

  InitKFoldCVMat(xs, this->xs);
  InitKFoldCVMat(ys, this->ys);

  // Do we need to shuffle the dataset?
  if (shuffle)
    Shuffle();
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>::KFoldCV(Base&& base,
                              const size_t k,
                              const MatType& xs,
                              const PredictionsType& ys,
                              const WeightsType& weights,
                              const bool shuffle) :
    base(std::move(base)),
    k(k)
{
  Base::AssertWeightsConsistency(xs, weights);

  InitKFoldCVMat(xs, this->xs);
  InitKFoldCVMat(ys, this->ys);
  InitKFoldCVMat(weights, this->weights);

  // Do we need to shuffle the dataset?
  if (shuffle)
    Shuffle();
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs>
double KFoldCV<MLAlgorithm,
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
MLAlgorithm& KFoldCV<MLAlgorithm,
                     Metric,
                     MatType,
                     PredictionsType,
                     WeightsType>::Model()
{
  if (modelPtr == nullptr)
    throw std::logic_error(
        "KFoldCV::Model(): attempted to access an uninitialized model");

  return *modelPtr;
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename DataType>
void KFoldCV<MLAlgorithm,
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

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename>
double KFoldCV<MLAlgorithm,
                Metric,
                MatType,
                PredictionsType,
                WeightsType>::TrainAndEvaluate(const MLAlgorithmArgs&... args)
{
  arma::vec evaluations(k);

  size_t numInvalidScores = 0;
  for (size_t i = 0; i < k; ++i)
  {
    MLAlgorithm&& model  = base.Train(GetTrainingSubset(xs, i),
        GetTrainingSubset(ys, i), args...);
    evaluations(i) = Metric::Evaluate(model, GetValidationSubset(xs, i),
        GetValidationSubset(ys, i));
    if (std::isnan(evaluations(i)) || std::isinf(evaluations(i)))
    {
      ++numInvalidScores;
      Log::Warn << "KFoldCV::TrainAndEvaluate(): fold " << i << " returned "
          << "a score of " << evaluations(i) << "; ignoring when computing "
          << "the average score." << std::endl;
    }
    if (i == k - 1)
      modelPtr.reset(new MLAlgorithm(std::move(model)));
  }

  if (numInvalidScores == k)
  {
    Log::Warn << "KFoldCV::TrainAndEvaluate(): all folds returned invalid "
        << "scores!  Returning 0.0 as overall score." << std::endl;
    return 0.0;
  }

  return arma::mean(evaluations.elem(arma::find_finite(evaluations)));
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename, typename>
double KFoldCV<MLAlgorithm,
                Metric,
                MatType,
                PredictionsType,
                WeightsType>::TrainAndEvaluate(const MLAlgorithmArgs&... args)
{
  arma::vec evaluations(k);

  for (size_t i = 0; i < k; ++i)
  {
    MLAlgorithm&& model = (weights.n_elem > 0) ?
        base.Train(GetTrainingSubset(xs, i), GetTrainingSubset(ys, i),
            GetTrainingSubset(weights, i), args...) :
        base.Train(GetTrainingSubset(xs, i), GetTrainingSubset(ys, i),
            args...);
    evaluations(i) = Metric::Evaluate(model, GetValidationSubset(xs, i),
        GetValidationSubset(ys, i));
    if (i == k - 1)
      modelPtr.reset(new MLAlgorithm(std::move(model)));
  }

  return arma::mean(evaluations);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<bool Enabled, typename>
void KFoldCV<MLAlgorithm,
             Metric,
             MatType,
             PredictionsType,
             WeightsType>::Shuffle()
{
  MatType xsOrig = xs.cols(0, (k - 1) * binSize + lastBinSize - 1);
  PredictionsType ysOrig = ys.cols(0, (k - 1) * binSize + lastBinSize - 1);

  // Now shuffle the data.
  ShuffleData(xsOrig, ysOrig, xsOrig, ysOrig);

  InitKFoldCVMat(xsOrig, xs);
  InitKFoldCVMat(ysOrig, ys);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<bool Enabled, typename, typename>
void KFoldCV<MLAlgorithm,
             Metric,
             MatType,
             PredictionsType,
             WeightsType>::Shuffle()
{
  MatType xsOrig = xs.cols(0, (k - 1) * binSize + lastBinSize - 1);
  PredictionsType ysOrig = ys.cols(0, (k - 1) * binSize + lastBinSize - 1);
  WeightsType weightsOrig;
  if (weights.n_elem > 0)
    weightsOrig = weights.cols(0, (k - 1) * binSize + lastBinSize - 1);

  // Now shuffle the data.
  if (weights.n_elem > 0)
    ShuffleData(xsOrig, ysOrig, weightsOrig, xsOrig, ysOrig, weightsOrig);
  else
    ShuffleData(xsOrig, ysOrig, xsOrig, ysOrig);

  InitKFoldCVMat(xsOrig, xs);
  InitKFoldCVMat(ysOrig, ys);
  if (weights.n_elem > 0)
    InitKFoldCVMat(weightsOrig, weights);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
size_t KFoldCV<MLAlgorithm,
               Metric,
               MatType,
               PredictionsType,
               WeightsType>::ValidationSubsetFirstCol(const size_t i)
{
  // Use as close to the beginning of the dataset as we can.
  return (i == 0) ? binSize * (k - 1) : binSize * (i - 1);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename ElementType>
arma::Mat<ElementType> KFoldCV<MLAlgorithm,
                               Metric,
                               MatType,
                               PredictionsType,
                               WeightsType>::GetTrainingSubset(
    arma::Mat<ElementType>& m,
    const size_t i)
{
  // If this is not the first fold, we have to handle it a little bit
  // differently, since the last fold may contain slightly more than 'binSize'
  // points.
  const size_t subsetSize = (i != 0) ? lastBinSize + (k - 2) * binSize :
      (k - 1) * binSize;

  return arma::Mat<ElementType>(m.colptr(binSize * i), m.n_rows, subsetSize,
      false, true);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename ElementType>
arma::Row<ElementType> KFoldCV<MLAlgorithm,
                               Metric,
                               MatType,
                               PredictionsType,
                               WeightsType>::GetTrainingSubset(
    arma::Row<ElementType>& r,
    const size_t i)
{
  // If this is not the first fold, we have to handle it a little bit
  // differently, since the last fold may contain slightly more than 'binSize'
  // points.
  const size_t subsetSize = (i != 0) ? lastBinSize + (k - 2) * binSize :
      (k - 1) * binSize;

  return arma::Row<ElementType>(r.colptr(binSize * i), subsetSize, false, true);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename ElementType>
arma::Mat<ElementType> KFoldCV<MLAlgorithm,
                               Metric,
                               MatType,
                               PredictionsType,
                               WeightsType>::GetValidationSubset(
    arma::Mat<ElementType>& m,
    const size_t i)
{
  const size_t subsetSize = (i == 0) ? lastBinSize : binSize;
  return arma::Mat<ElementType>(m.colptr(ValidationSubsetFirstCol(i)), m.n_rows,
      subsetSize, false, true);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename ElementType>
arma::Row<ElementType> KFoldCV<MLAlgorithm,
                               Metric,
                               MatType,
                               PredictionsType,
                               WeightsType>::GetValidationSubset(
    arma::Row<ElementType>& r,
    const size_t i)
{
  const size_t subsetSize = (i == 0) ? lastBinSize : binSize;
  return arma::Row<ElementType>(r.colptr(ValidationSubsetFirstCol(i)),
      subsetSize, false, true);
}

} // namespace mlpack

#endif
