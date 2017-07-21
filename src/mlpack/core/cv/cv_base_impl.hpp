/**
 * @file cv_base_impl.hpp
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

namespace mlpack {
namespace cv {

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MatInType, typename PredictionsInType>
CVBase<MLAlgorithm,
       MatType,
       PredictionsType,
       WeightsType>::CVBase(const MatInType&,
                            const PredictionsInType&) :
    isDatasetInfoPassed(false)
{
  static_assert(!MIE::TakesNumClasses,
      "The given MLAlgorithm requires the numClasses parameter");
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MatInType, typename PredictionsInType>
CVBase<MLAlgorithm,
       MatType,
       PredictionsType,
       WeightsType>::CVBase(const MatInType&,
                            const PredictionsInType&,
                            const size_t numClasses) :
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
template<typename MatInType, typename PredictionsInType>
CVBase<MLAlgorithm,
       MatType,
       PredictionsType,
       WeightsType>::CVBase(const MatInType&,
                            const data::DatasetInfo& datasetInfo,
                            const PredictionsInType&,
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
template<typename MatInType, typename PredictionsInType, typename WeightsInType>
CVBase<MLAlgorithm,
       MatType,
       PredictionsType,
       WeightsType>::CVBase(const MatInType&,
                            const PredictionsInType&,
                            const WeightsInType&) :
    isDatasetInfoPassed(false)
{
  static_assert(!MIE::TakesNumClasses,
      "The given MLAlgorithm requires the numClasses parameter");
  static_assert(MIE::SupportsWeights,
      "The given MLAlgorithm does not support weighted learning");
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MatInType, typename PredictionsInType, typename WeightsInType>
CVBase<MLAlgorithm,
       MatType,
       PredictionsType,
       WeightsType>::CVBase(const MatInType&,
                            const PredictionsInType&,
                            const size_t numClasses,
                            const WeightsInType&) :
    isDatasetInfoPassed(false),
    numClasses(numClasses)
{
  static_assert(MIE::TakesNumClasses,
      "The given MLAlgorithm does not take the numClasses parameter");
  static_assert(MIE::SupportsWeights,
      "The given MLAlgorithm does not support weighted learning");
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename MatInType, typename PredictionsInType, typename WeightsInType>
CVBase<MLAlgorithm,
       MatType,
       PredictionsType,
       WeightsType>::CVBase(const MatInType&,
                            const data::DatasetInfo& datasetInfo,
                            const PredictionsInType&,
                            const size_t numClasses,
                            const WeightsInType&) :
    datasetInfo(datasetInfo),
    isDatasetInfoPassed(true),
    numClasses(numClasses)
{
  static_assert(MIE::TakesNumClasses,
      "The given MLAlgorithm does not take the numClasses parameter");
  static_assert(MIE::TakesDatasetInfo,
      "The given MLAlgorithm does not accept a data::DatasetInfo parameter");
  static_assert(MIE::SupportsWeights,
      "The given MLAlgorithm does not support weighted learning");
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs>
std::unique_ptr<MLAlgorithm> CVBase<MLAlgorithm,
                                    MatType,
                                    PredictionsType,
                                    WeightsType>::Train(
    const MatType& xs,
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
std::unique_ptr<MLAlgorithm> CVBase<MLAlgorithm,
                                    MatType,
                                    PredictionsType,
                                    WeightsType>::Train(
    const MatType& xs,
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
  AssertSizeEquality(xs, ys);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
void CVBase<MLAlgorithm,
            MatType,
            PredictionsType,
            WeightsType>::AssertDataConsistency(const MatType& xs,
                                                const PredictionsType& ys,
                                                const WeightsType& weights)
{
  AssertSizeEquality(xs, ys);
  AssertWeightsSize(xs, weights);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
void CVBase<MLAlgorithm,
            MatType,
            PredictionsType,
            WeightsType>::AssertSizeEquality(const MatType& xs,
                                             const PredictionsType& ys)
{
  if (xs.n_cols != ys.n_cols)
  {
    std::ostringstream oss;
    oss << "CVBase::AssertSizeEquality(): number of data points (" << xs.n_cols
        << ") does not match number of predictions (" << ys.n_cols << ")!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
void CVBase<MLAlgorithm,
            MatType,
            PredictionsType,
            WeightsType>::AssertWeightsSize(const MatType& xs,
                                            const WeightsType& weights)
{
  if (weights.n_elem != xs.n_cols)
  {
    std::ostringstream oss;
    oss << "CVBase::AssertWeightsSize(): number of weights ("
        << weights.n_elem << ") does not match number of data points ("
        << xs.n_cols << ")!" << std::endl;
    throw std::invalid_argument(oss.str());
  }
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename>
std::unique_ptr<MLAlgorithm> CVBase<MLAlgorithm,
                                    MatType,
                                    PredictionsType,
                                    WeightsType>::TrainModel(
    const MatType& xs,
    const PredictionsType& ys,
    const MLAlgorithmArgs&... args)
{
  static_assert(
      std::is_constructible<MLAlgorithm, const MatType&, const PredictionsType&,
          MLAlgorithmArgs...>::value,
      "The given MLAlgorithm is not constructible from the passed arguments");

  return std::unique_ptr<MLAlgorithm>(new MLAlgorithm(xs, ys, args...));
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename, typename>
std::unique_ptr<MLAlgorithm> CVBase<MLAlgorithm,
                                    MatType,
                                    PredictionsType,
                                    WeightsType>::TrainModel(
    const MatType& xs,
    const PredictionsType& ys,
    const MLAlgorithmArgs&... args)
{
  static_assert(
      std::is_constructible<MLAlgorithm, const MatType&, const PredictionsType&,
          const size_t, MLAlgorithmArgs...>::value,
      "The given MLAlgorithm is not constructible from the passed arguments");

  return std::unique_ptr<MLAlgorithm>(
      new MLAlgorithm(xs, ys, numClasses, args...));
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename, typename,
    typename>
std::unique_ptr<MLAlgorithm> CVBase<MLAlgorithm,
                                    MatType,
                                    PredictionsType,
                                    WeightsType>::TrainModel(
    const MatType& xs,
    const PredictionsType& ys,
    const MLAlgorithmArgs&... args)
{
  static_assert(
      std::is_constructible<MLAlgorithm, const MatType&,
          const data::DatasetInfo, const PredictionsType&, const size_t,
              MLAlgorithmArgs...>::value,
      "The given MLAlgorithm is not constructible with a data::DatasetInfo "
      "parameter and the passed arguments");

  static const bool constructableWithoutDatasetInfo =
      std::is_constructible<MLAlgorithm, const MatType&, const PredictionsType&,
          const size_t, MLAlgorithmArgs...>::value;
  return TrainModel<constructableWithoutDatasetInfo>(xs, ys, args...);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename>
std::unique_ptr<MLAlgorithm> CVBase<MLAlgorithm,
                                    MatType,
                                    PredictionsType,
                                    WeightsType>::TrainModel(
    const MatType& xs,
    const PredictionsType& ys,
    const WeightsType& weights,
    const MLAlgorithmArgs&... args)
{
  static_assert(
      std::is_constructible<MLAlgorithm, const MatType&, const PredictionsType&,
          const WeightsType&, MLAlgorithmArgs...>::value,
      "The given MLAlgorithm is not constructible from the passed arguments");

  return std::unique_ptr<MLAlgorithm>(
      new MLAlgorithm(xs, ys, weights, args...));
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename, typename>
std::unique_ptr<MLAlgorithm> CVBase<MLAlgorithm,
                                    MatType,
                                    PredictionsType,
                                    WeightsType>::TrainModel(
    const MatType& xs,
    const PredictionsType& ys,
    const WeightsType& weights,
    const MLAlgorithmArgs&... args)
{
  static_assert(
      std::is_constructible<MLAlgorithm, const MatType&, const PredictionsType&,
          const size_t, const WeightsType&, MLAlgorithmArgs...>::value,
      "The given MLAlgorithm is not constructible from the passed arguments");

  return std::unique_ptr<MLAlgorithm>(
      new MLAlgorithm(xs, ys, numClasses, weights, args...));
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... MLAlgorithmArgs, bool Enabled, typename, typename,
    typename>
std::unique_ptr<MLAlgorithm> CVBase<MLAlgorithm,
                                    MatType,
                                    PredictionsType,
                                    WeightsType>::TrainModel(
    const MatType& xs,
    const PredictionsType& ys,
    const WeightsType& weights,
    const MLAlgorithmArgs&... args)
{
  static_assert(
      std::is_constructible<MLAlgorithm, const MatType&,
          const data::DatasetInfo, const PredictionsType&, const size_t,
              const WeightsType&, MLAlgorithmArgs...>::value,
      "The given MLAlgorithm is not constructible with a data::DatasetInfo "
      "parameter and the passed arguments");

  static const bool constructableWithoutDatasetInfo =
      std::is_constructible<MLAlgorithm, const MatType&, const PredictionsType&,
          const size_t, const WeightsType&, MLAlgorithmArgs...>::value;
  return TrainModel<constructableWithoutDatasetInfo>(xs, ys, weights, args...);
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<bool ConstructableWithoutDatasetInfo, typename... MLAlgorithmArgs,
    typename>
std::unique_ptr<MLAlgorithm> CVBase<MLAlgorithm,
                                    MatType,
                                    PredictionsType,
                                    WeightsType>::TrainModel(
    const MatType& xs,
    const PredictionsType& ys,
    const MLAlgorithmArgs&... args)
{
  if (isDatasetInfoPassed)
    return std::unique_ptr<MLAlgorithm>(
        new MLAlgorithm(xs, datasetInfo, ys, numClasses, args...));
  else
    return std::unique_ptr<MLAlgorithm>(
        new MLAlgorithm(xs, ys, numClasses, args...));
}

template<typename MLAlgorithm,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<bool ConstructableWithoutDatasetInfo, typename... MLAlgorithmArgs,
    typename, typename>
std::unique_ptr<MLAlgorithm> CVBase<MLAlgorithm,
                                    MatType,
                                    PredictionsType,
                                    WeightsType>::TrainModel(
    const MatType& xs,
    const PredictionsType& ys,
    const MLAlgorithmArgs&... args)
{
  if (!isDatasetInfoPassed)
    throw std::invalid_argument(
        "The given MLAlgorithm requires a data::DatasetInfo parameter");

  return std::unique_ptr<MLAlgorithm>(
      new MLAlgorithm(xs, datasetInfo, ys, numClasses, args...));
}

} // namespace cv
} // namespace mlpack

#endif
