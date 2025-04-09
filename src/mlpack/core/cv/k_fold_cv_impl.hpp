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
    KFoldCVBase<
        KFoldCV<MLAlgorithm, Metric, MatType, PredictionsType, WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>(k, xs, ys)
{
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
        WeightsType>::KFoldCV(const size_t k,
                              const MatType& xs,
                              const PredictionsType& ys,
                              const size_t numClasses,
                              const bool shuffle) :
    KFoldCVBase<
        KFoldCV<MLAlgorithm, Metric, MatType, PredictionsType, WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>(k, xs, ys, numClasses)
{
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
        WeightsType>::KFoldCV(const size_t k,
                              const MatType& xs,
                              const data::DatasetInfo& datasetInfo,
                              const PredictionsType& ys,
                              const size_t numClasses,
                              const bool shuffle) :
    KFoldCVBase<
        KFoldCV<MLAlgorithm, Metric, MatType, PredictionsType, WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>(k, xs, datasetInfo, ys, numClasses)
{
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
        WeightsType>::KFoldCV(const size_t k,
                              const MatType& xs,
                              const PredictionsType& ys,
                              const WeightsType& weights,
                              const bool shuffle) :
    KFoldCVBase<
        KFoldCV<MLAlgorithm, Metric, MatType, PredictionsType, WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>(k, xs, ys, weights)
{ 
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
        WeightsType>::KFoldCV(const size_t k,
                              const MatType& xs,
                              const PredictionsType& ys,
                              const size_t numClasses,
                              const WeightsType& weights,
                              const bool shuffle) :
    KFoldCVBase<
        KFoldCV<MLAlgorithm, Metric, MatType, PredictionsType, WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>(k, xs, ys, numClasses, weights)
{
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
        WeightsType>::KFoldCV(const size_t k,
                              const MatType& xs,
                              const data::DatasetInfo& datasetInfo,
                              const PredictionsType& ys,
                              const size_t numClasses,
                              const WeightsType& weights,
                              const bool shuffle) :
    KFoldCVBase<
        KFoldCV<MLAlgorithm, Metric, MatType, PredictionsType, WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>(k, xs, datasetInfo, ys, numClasses, weights)
{ 
  // Do we need to shuffle the dataset?
  if (shuffle)
    Shuffle();
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
  const size_t index = (this->k - 1) * this->binSize + this->lastBinSize - 1;
  MatType xsOrig = this->xs.cols(0, index);
  PredictionsType ysOrig = this->ys.cols(0, index);

  // Now shuffle the data.
  ShuffleData(xsOrig, ysOrig, xsOrig, ysOrig);

  this->InitKFoldCVMat(xsOrig, this->xs);
  this->InitKFoldCVMat(ysOrig, this->ys);
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
  const size_t index = (this->k - 1) * this->binSize + this->lastBinSize - 1;
  MatType xsOrig = this->xs.cols(0, index);
  PredictionsType ysOrig = this->ys.cols(0, index);
  WeightsType weightsOrig;
  if (this->weights.n_elem > 0)
    weightsOrig = this->weights.cols(0, index);

  // Now shuffle the data.
  if (this->weights.n_elem > 0)
    ShuffleData(xsOrig, ysOrig, weightsOrig, xsOrig, ysOrig, weightsOrig);
  else
    ShuffleData(xsOrig, ysOrig, xsOrig, ysOrig);

  this->InitKFoldCVMat(xsOrig, this->xs);
  this->InitKFoldCVMat(ysOrig, this->ys);
  if (this->weights.n_elem > 0)
    this->InitKFoldCVMat(weightsOrig, this->weights);
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
  return (i == 0) ? this->binSize * (this->k - 1) : this->binSize * (i - 1);
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
  const size_t subsetSize = (i != 0) ?
      this->lastBinSize + (this->k - 2) * this->binSize :
      (this->k - 1) * this->binSize;

  return arma::Mat<ElementType>(m.colptr(this->binSize * i), m.n_rows, subsetSize,
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
  const size_t subsetSize = (i != 0) ? 
      this->lastBinSize + (this->k - 2) * this->binSize :
      (this->k - 1) * this->binSize;

  return arma::Row<ElementType>(r.colptr(this->binSize * i), subsetSize, false, true);
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
  const size_t subsetSize = (i == 0) ? this->lastBinSize : this->binSize;
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
  const size_t subsetSize = (i == 0) ? this->lastBinSize : this->binSize;
  return arma::Row<ElementType>(r.colptr(ValidationSubsetFirstCol(i)),
      subsetSize, false, true);
}

} // namespace mlpack

#endif
