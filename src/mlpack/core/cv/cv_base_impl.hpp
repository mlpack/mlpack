/**
 * @file core/cv/cv_base_impl.hpp
 * @author Kirill Mishchenko
 *
 * The implementation of the class CVBase.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_CV_BASE_IMPL_HPP
#define MLPACK_CORE_CV_CV_BASE_IMPL_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
CVBase<MLAlgorithm,
       MatType,
       PredictionsType,
       WeightsType>::CVBase() :
    isDatasetInfoPassed(false),
    numClasses(0)
{
  static_assert(!MIE::TakesNumClasses,
      "The given MLAlgorithm requires the numClasses parameter; "
      "make sure that you pass numClasses with type size_t!");
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
CVBase<MLAlgorithm,
       MatType,
       PredictionsType,
       WeightsType>::CVBase(const size_t numClasses) :
    isDatasetInfoPassed(false),
    numClasses(numClasses)
{
  static_assert(MIE::TakesNumClasses,
      "The given MLAlgorithm does not take the numClasses parameter");
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
CVBase<MLAlgorithm,
       MatType,
       PredictionsType,
       WeightsType>::CVBase(const data::DatasetInfo& datasetInfo,
                            const size_t numClasses) :
    datasetInfo(datasetInfo),
    isDatasetInfoPassed(true),
    numClasses(numClasses)
{
  static_assert(MIE::TakesNumClasses,
      "The given MLAlgorithm does not take the numClasses parameter");
  static_assert(MIE::TakesDatasetInfo,
      "The given MLAlgorithm does not accept a data::DatasetInfo parameter");
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs>
MLAlgorithm CVBase<MLAlgorithm,
                   MatType,
                   PredictionsType,
                   WeightsType>::Train(const MatType& xs,
                                       const PredictionsType& ys,
                                       const MLAlgorithmArgs&... args)
{
  return TrainModel(xs, ys, args...);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs>
MLAlgorithm CVBase<MLAlgorithm,
                   MatType,
                   PredictionsType,
                   WeightsType>::Train(const MatType& xs,
                                       const PredictionsType& ys,
                                       const WeightsType& weights,
                                       const MLAlgorithmArgs&... args)
{
  return TrainModel(xs, ys, weights, args...);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
void CVBase<MLAlgorithm,
            MatType,
            PredictionsType,
            WeightsType>::AssertDataConsistency(const MatType& xs,
                                                const PredictionsType& ys)
{
  util::CheckSameSizes(xs, (size_t) ys.n_cols,
      "CVBase::AssertDataConsistency()", "predictions");
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
void CVBase<MLAlgorithm,
            MatType,
            PredictionsType,
            WeightsType>::AssertWeightsConsistency(const MatType& xs,
                                                   const WeightsType& weights)
{
  static_assert(MIE::SupportsWeights,
      "The given MLAlgorithm does not support weighted learning");

  util::CheckSameSizes(xs, weights, "CVBase::AssertWeightsConsistency()",
      "weights");
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename>
MLAlgorithm CVBase<MLAlgorithm,
                   MatType,
                   PredictionsType,
                   WeightsType>::TrainModel(const MatType& xs,
                                            const PredictionsType& ys,
                                            const MLAlgorithmArgs&... args)
{
  static_assert(
      std::is_constructible_v<MLAlgorithm,
          const MatType&, const PredictionsType&, MLAlgorithmArgs...>,
      "The given MLAlgorithm is not constructible from the passed arguments");

  return MLAlgorithm(xs, ys, args...);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename, typename>
MLAlgorithm CVBase<MLAlgorithm,
                   MatType,
                   PredictionsType,
                   WeightsType>::TrainModel(const MatType& xs,
                                            const PredictionsType& ys,
                                            const MLAlgorithmArgs&... args)
{
  static_assert(
      std::is_constructible_v<MLAlgorithm,
          const MatType&, const PredictionsType&,
          const size_t, MLAlgorithmArgs...>,
      "The given MLAlgorithm is not constructible from the passed arguments");

  return MLAlgorithm(xs, ys, numClasses, args...);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename, typename,
    typename>
MLAlgorithm CVBase<MLAlgorithm,
                   MatType,
                   PredictionsType,
                   WeightsType>::TrainModel(const MatType& xs,
                                            const PredictionsType& ys,
                                            const MLAlgorithmArgs&... args)
{
  static_assert(
      std::is_constructible_v<MLAlgorithm, const MatType&,
          const data::DatasetInfo, const PredictionsType&, const size_t,
              MLAlgorithmArgs...>,
      "The given MLAlgorithm is not constructible with a data::DatasetInfo "
      "parameter and the passed arguments");

  static const bool constructableWithoutDatasetInfo =
      std::is_constructible_v<MLAlgorithm,
          const MatType&, const PredictionsType&,
          const size_t, MLAlgorithmArgs...>;
  return TrainModel<constructableWithoutDatasetInfo>(xs, ys, args...);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename>
MLAlgorithm CVBase<MLAlgorithm,
                   MatType,
                   PredictionsType,
                   WeightsType>::TrainModel(const MatType& xs,
                                            const PredictionsType& ys,
                                            const WeightsType& weights,
                                            const MLAlgorithmArgs&... args)
{
  static_assert(
      std::is_constructible_v<MLAlgorithm,
          const MatType&, const PredictionsType&,
          const WeightsType&, MLAlgorithmArgs...>,
      "The given MLAlgorithm is not constructible from the passed arguments");

  return MLAlgorithm(xs, ys, weights, args...);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename, typename>
MLAlgorithm CVBase<MLAlgorithm,
                   MatType,
                   PredictionsType,
                   WeightsType>::TrainModel(const MatType& xs,
                                            const PredictionsType& ys,
                                            const WeightsType& weights,
                                            const MLAlgorithmArgs&... args)
{
  static_assert(
      std::is_constructible_v<MLAlgorithm,
          const MatType&, const PredictionsType&,
          const size_t, const WeightsType&, MLAlgorithmArgs...>,
      "The given MLAlgorithm is not constructible from the passed arguments");

  return MLAlgorithm(xs, ys, numClasses, weights, args...);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename, typename,
    typename>
MLAlgorithm CVBase<MLAlgorithm,
                  MatType,
                  PredictionsType,
                  WeightsType>::TrainModel(const MatType& xs,
                                           const PredictionsType& ys,
                                           const WeightsType& weights,
                                           const MLAlgorithmArgs&... args)
{
  static_assert(
      std::is_constructible_v<MLAlgorithm, const MatType&,
          const data::DatasetInfo, const PredictionsType&, const size_t,
              const WeightsType&, MLAlgorithmArgs...>,
      "The given MLAlgorithm is not constructible with a data::DatasetInfo "
      "parameter and the passed arguments");

  static const bool constructableWithoutDatasetInfo =
      std::is_constructible_v<MLAlgorithm,
          const MatType&, const PredictionsType&,
          const size_t, const WeightsType&, MLAlgorithmArgs...>;
  return TrainModel<constructableWithoutDatasetInfo>(xs, ys, weights, args...);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<bool ConstructableWithoutDatasetInfo, typename... MLAlgorithmArgs,
    typename>
MLAlgorithm CVBase<MLAlgorithm,
                   MatType,
                   PredictionsType,
                   WeightsType>::TrainModel(const MatType& xs,
                                            const PredictionsType& ys,
                                            const MLAlgorithmArgs&... args)
{
  if (isDatasetInfoPassed)
    return MLAlgorithm(xs, datasetInfo, ys, numClasses, args...);
  else
    return MLAlgorithm(xs, ys, numClasses, args...);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<bool ConstructableWithoutDatasetInfo, typename... MLAlgorithmArgs,
    typename, typename>
MLAlgorithm CVBase<MLAlgorithm,
                   MatType,
                   PredictionsType,
                   WeightsType>::TrainModel(const MatType& xs,
                                            const PredictionsType& ys,
                                            const MLAlgorithmArgs&... args)
{
  if (!isDatasetInfoPassed)
    throw std::invalid_argument(
        "The given MLAlgorithm requires a data::DatasetInfo parameter");

  return MLAlgorithm(xs, datasetInfo, ys, numClasses, args...);
}

} // namespace mlpack

#endif
