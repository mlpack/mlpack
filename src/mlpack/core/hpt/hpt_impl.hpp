/**
 * @file hpt_impl.hpp
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
namespace hpt {

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
    cv(args...) {}

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
  arma::mat bestParameters;
  const auto argsTuple = std::tie(args...);

  InitGridSearch<0>(argsTuple);
  InitCVFunctionAndOptimize<0>(argsTuple, bestParameters);

  return VectorToTuple<TupleOfHyperParameters<Args...>, 0>(bestParameters);
}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<size_t I, typename ArgsTuple, typename... Collections, typename>
void HyperParameterTuner<MLAlgorithm,
                         Metric,
                         CV,
                         Optimizer,
                         MatType,
                         PredictionsType,
                         WeightsType>::InitGridSearch(
    const ArgsTuple& /* args */,
    Collections... collections)
{
  optimizer = Optimizer(collections...);
}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<size_t I, class ArgsTuple, class... Collections, typename, typename>
void HyperParameterTuner<MLAlgorithm,
                         Metric,
                         CV,
                         Optimizer,
                         MatType,
                         PredictionsType,
                         WeightsType>::InitGridSearch(
    const ArgsTuple& args,
    Collections... collections)
{
  InitGridSearch<I + 1>(args, collections...);
}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<size_t I, class ArgsTuple, class... Collections, class, class, class>
void HyperParameterTuner<MLAlgorithm,
                         Metric,
                         CV,
                         Optimizer,
                         MatType,
                         PredictionsType,
                         WeightsType>::InitGridSearch(
    const ArgsTuple& args,
    Collections... collections)
{
  InitGridSearch<I + 1>(args, collections..., std::get<I>(args));
}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<size_t I, typename ArgsTuple, typename... BoundArgs, typename>
void HyperParameterTuner<MLAlgorithm,
                         Metric,
                         CV,
                         Optimizer,
                         MatType,
                         PredictionsType,
                         WeightsType>::InitCVFunctionAndOptimize(
    const ArgsTuple& /* args */,
    arma::mat& bestParams,
    BoundArgs... boundArgs)
{
  static const size_t totalArgs = std::tuple_size<ArgsTuple>::value;

  CVFunction<CVType, MLAlgorithm, totalArgs, BoundArgs...>
      cvFunction(cv, boundArgs...);
  bestObjective = Metric::NeedsMinimization?
      optimizer.Optimize(cvFunction, bestParams) :
      -optimizer.Optimize(cvFunction, bestParams);
  bestModel = std::move(cvFunction.BestModel());
}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<size_t I, class ArgsTuple, class... BoundArgs, class, class>
void HyperParameterTuner<MLAlgorithm,
                         Metric,
                         CV,
                         Optimizer,
                         MatType,
                         PredictionsType,
                         WeightsType>::InitCVFunctionAndOptimize(
    const ArgsTuple& args,
    arma::mat& bestParams,
    BoundArgs... boundArgs)
{
  using PreBoundArgT = typename std::remove_reference<
      typename std::tuple_element<I, ArgsTuple>::type>::type;
  using BoundArgT = BoundArg<typename PreBoundArgT::Type, I>;

  InitCVFunctionAndOptimize<I + 1>(args, bestParams, boundArgs...,
       BoundArgT{std::get<I>(args).value});
}

template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer,
         typename MatType,
         typename PredictionsType,
         typename WeightsType>
template<size_t I, class ArgsTuple, class... BoundArgs, class, class, class>
void HyperParameterTuner<MLAlgorithm,
                         Metric,
                         CV,
                         Optimizer,
                         MatType,
                         PredictionsType,
                         WeightsType>::InitCVFunctionAndOptimize(
    const ArgsTuple& args,
    arma::mat& bestParams,
    BoundArgs... boundArgs)
{
  InitCVFunctionAndOptimize<I + 1>(args, bestParams, boundArgs...);
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
template<typename TupleType, size_t, typename... Args, typename, typename>
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

} // namespace hpt
} // namespace mlpack

#endif
