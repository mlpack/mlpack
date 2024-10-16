/**
 * @file core/hpt/hpt_impl.hpp
 * @author Kirill Mishchenko
 *
 * Implementation of hyper-parameter tuning.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_HPT_HPT_IMPL_HPP
#define MLPACK_CORE_HPT_HPT_IMPL_HPP

#include <mlpack/core.hpp>

namespace mlpack {

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... CVArgs>
HyperParameterTuner<MLAlgorithm,
                    Metric,
                    CV,
                    Optimizer,
                    MatType,
                    PredictionsType,
                    WeightsType>::HyperParameterTuner(const CVArgs&... args) :
    cv(args...), relativeDelta(0.01), minDelta(1e-10) {}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename... Args>
TupleOfHyperParameters<Args...> HyperParameterTuner<MLAlgorithm,
                                                    Metric,
                                                    CV,
                                                    Optimizer,
                                                    MatType,
                                                    PredictionsType,
                                                    WeightsType>::Optimize(
    const Args&... args)
{
  static const size_t numberOfParametersToOptimize =
      std::tuple_size<TupleOfHyperParameters<Args...>>::value;
  data::IncrementPolicy policy(true);
  data::DatasetMapper<data::IncrementPolicy, double> datasetInfo(policy,
      numberOfParametersToOptimize);

  arma::mat bestParameters(numberOfParametersToOptimize, 1);
  const auto argsTuple = std::tie(args...);

  InitAndOptimize<0>(argsTuple, bestParameters, datasetInfo);

  for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
  {
    if (datasetInfo.Type(i) == data::Datatype::categorical)
      bestParameters[i] = datasetInfo.UnmapString(bestParameters[i], i);
  }

  return VectorToTuple<TupleOfHyperParameters<Args...>, 0>(bestParameters);
}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<size_t I, typename ArgsTuple, typename... FixedArgs, typename>
void HyperParameterTuner<MLAlgorithm,
                         Metric,
                         CV,
                         Optimizer,
                         MatType,
                         PredictionsType,
                         WeightsType>::InitAndOptimize(
    const ArgsTuple& /* args */,
    arma::mat& bestParams,
    data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
    FixedArgs... fixedArgs)
{
  static const size_t totalArgs = std::tuple_size<ArgsTuple>::value;

  std::vector<bool> categoricalDimensions(datasetInfo.Dimensionality());
  arma::Row<size_t> numCategories(datasetInfo.Dimensionality());
  for (size_t d = 0; d < datasetInfo.Dimensionality(); d++)
  {
    numCategories[d] = datasetInfo.NumMappings(d);
    categoricalDimensions[d] = datasetInfo.Type(d) ==
        mlpack::data::Datatype::categorical;
  }

  CVFunction<CVType, MLAlgorithm, totalArgs, FixedArgs...>
      cvFunction(cv, datasetInfo, relativeDelta, minDelta, fixedArgs...);
  bestObjective = Metric::NeedsMinimization ? optimizer.Optimize(cvFunction,
      bestParams, categoricalDimensions, numCategories) :
      -optimizer.Optimize(cvFunction, bestParams, categoricalDimensions,
      numCategories);
      bestModel = std::move(cvFunction.BestModel());
}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<size_t I, class ArgsTuple, class... FixedArgs, class, class>
void HyperParameterTuner<MLAlgorithm,
                         Metric,
                         CV,
                         Optimizer,
                         MatType,
                         PredictionsType,
                         WeightsType>::InitAndOptimize(
    const ArgsTuple& args,
    arma::mat& bestParams,
    data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
    FixedArgs... fixedArgs)
{
  using PreFixedArgT = std::remove_reference_t<
      std::tuple_element_t<I, ArgsTuple>>;
  using FixedArgT = FixedArg<typename PreFixedArgT::Type, I>;

  InitAndOptimize<I + 1>(args, bestParams, datasetInfo, fixedArgs...,
       FixedArgT{std::get<I>(args).value});
}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<size_t I, class ArgsTuple, class... FixedArgs, class, class, class>
void HyperParameterTuner<MLAlgorithm,
                         Metric,
                         CV,
                         Optimizer,
                         MatType,
                         PredictionsType,
                         WeightsType>::InitAndOptimize(
    const ArgsTuple& args,
    arma::mat& bestParams,
    data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
    FixedArgs... fixedArgs)
{
  static const size_t dimension =
      I - std::tuple_size<std::tuple<FixedArgs...>>::value;
  datasetInfo.Type(dimension) = data::Datatype::numeric;
  bestParams(dimension) = std::get<I>(args);

  InitAndOptimize<I + 1>(args, bestParams, datasetInfo, fixedArgs...);
}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<size_t I, class ArgsTuple, class... FixedArgs, class, class, class,
    class>
void HyperParameterTuner<MLAlgorithm,
                         Metric,
                         CV,
                         Optimizer,
                         MatType,
                         PredictionsType,
                         WeightsType>::InitAndOptimize(
    const ArgsTuple& args,
    arma::mat& bestParams,
    data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
    FixedArgs... fixedArgs)
{
  static const size_t dimension =
      I - std::tuple_size<std::tuple<FixedArgs...>>::value;
  for (auto value : std::get<I>(args))
    datasetInfo.MapString<size_t>(value, dimension);

  if (datasetInfo.NumMappings(dimension) == 0)
  {
      std::ostringstream oss;
      oss << "HyperParameterTuner::Optimize(): the collection passed as the "
          << "argument " << I + 1 << " is empty" << std::endl;
      throw std::invalid_argument(oss.str());
  }

  InitAndOptimize<I + 1>(args, bestParams, datasetInfo, fixedArgs...);
}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename TupleType, size_t I, typename... Args, typename>
TupleType HyperParameterTuner<MLAlgorithm,
                           Metric,
                           CV,
                           Optimizer,
                           MatType,
                           PredictionsType,
                           WeightsType>::VectorToTuple(
    const arma::vec& vector, const Args&... args)
{
  return VectorToTuple<TupleType, I + 1>(vector, args..., vector(I));
}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<typename TupleType, size_t I, typename... Args, typename, typename>
TupleType HyperParameterTuner<MLAlgorithm,
                           Metric,
                           CV,
                           Optimizer,
                           MatType,
                           PredictionsType,
                           WeightsType>::VectorToTuple(
    const arma::vec& /* vector */, const Args&... args)
{
  return TupleType(args...);
}

} // namespace mlpack

#endif
