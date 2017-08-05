/**
 * @file simple_cv_impl.hpp
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
namespace cv {

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... CVBaseArgs>
KFoldCV<MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>::KFoldCV(const size_t k,
                              const CVBaseArgs&... args) :
    Base(args...), k(k)
{
  if (k < 2)
    throw std::invalid_argument("KFoldCV: k should not be less than 2");

  Init(Base::ExtractDataArgs(args...));
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
template<typename DataArgsTupleT,
         typename>
void KFoldCV<MLAlgorithm,
             Metric,
             MatType,
             PredictionsType,
             WeightsType>::Init(const DataArgsTupleT& dataArgsTuple)
{
  InitKFoldCVMat(std::get<0>(dataArgsTuple), xs);
  InitKFoldCVMat(std::get<1>(dataArgsTuple), ys);

  Base::AssertDataConsistency(xs, ys);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename DataArgsTupleT,
         typename,
         typename>
void KFoldCV<MLAlgorithm,
             Metric,
             MatType,
             PredictionsType,
             WeightsType>::Init(const DataArgsTupleT& dataArgsTuple)
{
  InitKFoldCVMat(std::get<0>(dataArgsTuple), xs);
  InitKFoldCVMat(std::get<1>(dataArgsTuple), ys);
  InitKFoldCVMat(std::get<2>(dataArgsTuple), weights);

  Base::AssertDataConsistency(xs, ys, weights);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename SourceType, typename DestinationType>
void KFoldCV<MLAlgorithm,
             Metric,
             MatType,
             PredictionsType,
             WeightsType>::InitKFoldCVMat(const SourceType& source,
                                          DestinationType& destination)
{
  const DestinationType& sourceAsDT = source;

  binSize = sourceAsDT.n_cols / k;
  trainingSubsetSize = binSize * (k - 1);

  destination = arma::join_rows(sourceAsDT,
      sourceAsDT.cols(0, trainingSubsetSize - binSize - 1));
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool, typename>
double KFoldCV<MLAlgorithm,
                Metric,
                MatType,
                PredictionsType,
                WeightsType>::TrainAndEvaluate(const MLAlgorithmArgs&... args)
{
  arma::vec evaluations(k);

  for (size_t i = 0; i < k; ++i)
  {
    modelPtr = this->Train(GetTrainingSubset(xs, i), GetTrainingSubset(ys, i),
        args...);
    evaluations(i) = Metric::Evaluate(*modelPtr, GetValidationSubset(xs, i),
        GetValidationSubset(ys, i));
  }

  return arma::mean(evaluations);
}

template<typename MLAlgorithm,
         typename Metric,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool, typename, typename>
double KFoldCV<MLAlgorithm,
                Metric,
                MatType,
                PredictionsType,
                WeightsType>::TrainAndEvaluate(const MLAlgorithmArgs&... args)
{
  arma::vec evaluations(k);

  for (size_t i = 0; i < k; ++i)
  {
    if (weights.n_elem > 0)
      modelPtr = this->Train(GetTrainingSubset(xs, i), GetTrainingSubset(ys, i),
          GetTrainingSubset(weights, i), args...);
    else
      modelPtr = this->Train(GetTrainingSubset(xs, i), GetTrainingSubset(ys, i),
          args...);
    evaluations(i) = Metric::Evaluate(*modelPtr, GetValidationSubset(xs, i),
        GetValidationSubset(ys, i));
  }

  return arma::mean(evaluations);
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
  return i < k - 1? binSize * i + trainingSubsetSize : binSize * (i - 1);
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
  return arma::Mat<ElementType>(m.colptr(binSize * i), m.n_rows,
      trainingSubsetSize, false, true);
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
  return arma::Row<ElementType>(r.colptr(binSize * i), trainingSubsetSize,
      false, true);
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
  return arma::Mat<ElementType>(m.colptr(ValidationSubsetFirstCol(i)), m.n_rows,
      binSize, false, true);
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
  return arma::Row<ElementType>(r.colptr(ValidationSubsetFirstCol(i)), binSize,
      false, true);
}

} // namespace cv
} // namespace mlpack

#endif
