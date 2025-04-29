/**
 * @file
 * The implementation of k-fold cross-validation.
 * 
 * @authors Felix Patschkowski
 *
 * @copyright mlpack is free software; you may redistribute it and/or modify
 * it under the terms of the 3-clause BSD license.  You should have received a
 * copy of the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_PURGED_K_FOLD_CV_IMPL_HPP
#define MLPACK_CORE_CV_PURGED_K_FOLD_CV_IMPL_HPP

namespace mlpack {

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::PurgedKFoldCV(
        size_t k,
        const MatType& xs,
        const PredictionsType& ys,
        double embargoPercentage,
        const MatType& intervals) :
  KFoldCVBase<
  PurgedKFoldCV<MLAlgorithm, Metric, MatType, PredictionsType, WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>(k, xs, ys),
  embargoPercentage(embargoPercentage),
  intervals(intervals)
{
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::PurgedKFoldCV(
        size_t k,
        const MatType& xs,
        const PredictionsType& ys,
        const size_t numClasses,
        double embargoPercentage,
        const MatType& intervals) :
  KFoldCVBase<
        PurgedKFoldCV<MLAlgorithm,
                      Metric,
                      MatType,
                      PredictionsType,
                      WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>(k, xs, ys, numClasses),
  embargoPercentage(embargoPercentage),
  intervals(intervals)
{
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
PurgedKFoldCV<MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::PurgedKFoldCV(
        size_t k,
        const MatType& xs,
        const data::DatasetInfo& datasetInfo,
        const PredictionsType& ys,
        const size_t numClasses,
        double embargoPercentage,
        const MatType& intervals) :
  KFoldCVBase<
        PurgedKFoldCV<MLAlgorithm,
                      Metric,
                      MatType,
                      PredictionsType,
                      WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>(k, xs, datasetInfo, ys, numClasses),
  embargoPercentage(embargoPercentage),
  intervals(intervals)
{
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::PurgedKFoldCV(
        size_t k,
        const MatType& xs,
        const PredictionsType& ys,
        const WeightsType& weights,
        double embargoPercentage,
        const MatType& intervals) :
  KFoldCVBase<
        PurgedKFoldCV<MLAlgorithm,
                      Metric,
                      MatType,
                      PredictionsType,
                      WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>(k, xs, ys, weights),
  embargoPercentage(embargoPercentage),
  intervals(intervals)
{
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::PurgedKFoldCV(
        size_t k,
        const MatType& xs,
        const PredictionsType& ys,
        const size_t numClasses,
        const WeightsType& weights,
        double embargoPercentage,
        const MatType& intervals) :
  KFoldCVBase<
        PurgedKFoldCV<MLAlgorithm,
                      Metric,
                      MatType,
                      PredictionsType,
                      WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>(k, xs, ys, numClasses, weights),
  embargoPercentage(embargoPercentage),
  intervals(intervals)
{
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::PurgedKFoldCV(
        size_t k,
        const MatType& xs,
        const data::DatasetInfo& datasetInfo,
        const PredictionsType& ys,
        const size_t numClasses,
        const WeightsType& weights,
        double embargoPercentage,
        const MatType& intervals) :
  KFoldCVBase<
        PurgedKFoldCV<MLAlgorithm,
                      Metric,
                      MatType,
                      PredictionsType,
                      WeightsType>,
        MLAlgorithm,
        Metric,
        MatType,
        PredictionsType,
        WeightsType>(k, xs, datasetInfo, ys, numClasses, weights),
  embargoPercentage(embargoPercentage),
  intervals(intervals)
{
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
size_t PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::ValidationSubsetFirstCol(size_t i) const
{
  assert(i < this->k);

  return (this->k - i - 1) * this->binSize;
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
template<typename ElementType>
arma::Mat<ElementType> PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::GetTrainingSubset(const arma::Mat<ElementType>& m, size_t i)
{
  return m.cols(GetTrainingSubsetCols(i));
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
template<typename ElementType>
arma::Row<ElementType> PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::GetTrainingSubset(const arma::Row<ElementType>& r, size_t i)
{
  return r.cols(GetTrainingSubsetCols(i));
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
template<typename ElementType>
arma::Mat<ElementType> PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::GetValidationSubset(const arma::Mat<ElementType>& m, size_t i)
{
  return m.cols(GetValidationSubsetCols(i));
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
template<typename ElementType>
arma::Row<ElementType> PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::GetValidationSubset(const arma::Row<ElementType>& r, size_t i)
{
  return r.cols(GetValidationSubsetCols(i));
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
void PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::CheckState() const
{
  if (intervals.n_rows != 2)
    throw std::invalid_argument(
        "PurgedKFoldCV::CheckState(): "
        "intervals must be a 2 x m matrix!");

  for (arma::uword j(0); j < intervals.n_cols; ++j)
    if (intervals(0, j) > intervals(1, j))
      throw std::invalid_argument(
          "PurgedKFoldCV::CheckState(): "
          "start must not be after end!");
    else if (intervals(0, j) < 0 || intervals(0, j) > this->xs.n_cols)
      throw std::invalid_argument(
          "PurgedKFoldCV::CheckState(): "
          "start is outside of the dataset!");
    else if (intervals(1, j) < 0 || intervals(1, j) > this->xs.n_cols)
      throw std::invalid_argument(
          "PurgedKFoldCV::CheckState(): "
          "end is outside of the dataset!");
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
arma::span PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::GetValidationSubsetCols(size_t i) const
{
  const size_t firstCol(ValidationSubsetFirstCol(i));

  return arma::span(firstCol,
      std::min(firstCol + this->binSize,
          (this->k - 1) * this->binSize + this->lastBinSize) - 1);
}

template<
    typename MLAlgorithm,
    typename Metric,
    typename MatType,
    typename PredictionsType,
    typename WeightsType>
arma::uvec PurgedKFoldCV<
    MLAlgorithm,
    Metric,
    MatType,
    PredictionsType,
    WeightsType>::GetTrainingSubsetCols(size_t i) const
{
  const arma::span  validationSubsetCols(GetValidationSubsetCols(i));
  const arma::uword datasetSize(this->xs.n_cols);
  const double      h(std::round(datasetSize * embargoPercentage));
  arma::uvec        trainingSubset(datasetSize);
  arma::uword       trainingSubsetSize(0);

  // Purge from the training set before the validation set.
  for (arma::uword j(0); j < validationSubsetCols.a; ++j)
    if (intervals(1, j) < validationSubsetCols.a)
      trainingSubset(trainingSubsetSize++) = j;

  // Extend the validation set's data points' duration temporarily.
  // Then purge from the training set after the validation set.
  const arma::uword trainingSubsetFirstCol(
      intervals(1, validationSubsetCols).max() + h + 1);

  if (trainingSubsetFirstCol < datasetSize) {
    const arma::uword n(datasetSize - trainingSubsetFirstCol);

    trainingSubset.rows(trainingSubsetSize, trainingSubsetSize + n - 1) =
        arma::linspace<arma::uvec>(
            trainingSubsetFirstCol,
            datasetSize - 1,
            n);
    trainingSubsetSize += n;
  }

  return trainingSubset.rows(0, trainingSubsetSize - 1);
}

} // namespace mlpack

#endif
