/**
 * @file core/hpt/cv_function_impl.hpp
 * @author Kirill Mishchenko
 *
 * The implementation of the class CVFunction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_HPT_CV_FUNCTION_IMPL_HPP
#define MLPACK_CORE_HPT_CV_FUNCTION_IMPL_HPP

namespace mlpack {

template<typename CVType,
         typename MLAlgorithm,
         size_t TotalArgs,
         typename... BoundArgs>
template<size_t BoundArgIndex, size_t ParamIndex>
struct CVFunction<CVType, MLAlgorithm, TotalArgs, BoundArgs...>::UseBoundArg<
    BoundArgIndex, ParamIndex, true>
{
  using BoundArgType = typename
      std::tuple_element<BoundArgIndex, std::tuple<BoundArgs...>>::type;

  static const bool value = BoundArgType::index == BoundArgIndex + ParamIndex;
};

template<typename CVType,
         typename MLAlgorithm,
         size_t TotalArgs,
         typename... BoundArgs>
template<size_t BoundArgIndex, size_t ParamIndex>
struct CVFunction<CVType, MLAlgorithm, TotalArgs, BoundArgs...>::UseBoundArg<
    BoundArgIndex, ParamIndex, false>
{
  static const bool value = false;
};

template<typename CVType,
         typename MLAlgorithm,
         size_t TotalArgs,
         typename... BoundArgs>
CVFunction<CVType, MLAlgorithm, TotalArgs, BoundArgs...>::CVFunction(
    CVType& cv,
    data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
    const double relativeDelta,
    const double minDelta,
    const BoundArgs&... args) :
    cv(cv),
    datasetInfo(datasetInfo),
    boundArgs(args...),
    bestObjective(std::numeric_limits<double>::max()),
    relativeDelta(relativeDelta),
    minDelta(minDelta)
{ /* Nothing left to do. */ }

template<typename CVType,
         typename MLAlgorithm,
         size_t TotalArgs,
         typename... BoundArgs>
double CVFunction<CVType, MLAlgorithm, TotalArgs, BoundArgs...>::Evaluate(
    const arma::mat& parameters)
{
  return Evaluate<0, 0>(parameters);
}

template<typename CVType,
         typename MLAlgorithm,
         size_t TotalArgs,
         typename... BoundArgs>
void CVFunction<CVType, MLAlgorithm, TotalArgs, BoundArgs...>::Gradient(
    const arma::mat& parameters,
    arma::mat& gradient)
{
  gradient = arma::mat(arma::size(parameters));
  arma::mat increasedParameters = parameters;
  double originalParametersEvaluation = Evaluate(parameters);
  for (size_t i = 0; i < parameters.n_rows; ++i)
  {
    double delta = std::max(std::abs(parameters(i)) * relativeDelta, minDelta);
    increasedParameters(i) += delta;
    gradient(i) =
        (Evaluate(increasedParameters) - originalParametersEvaluation) / delta;
    increasedParameters(i) = parameters(i);
  }
}

template<typename CVType,
         typename MLAlgorithm,
         size_t TotalArgs,
         typename... BoundArgs>
template<size_t BoundArgIndex,
         size_t ParamIndex,
         typename... Args,
         typename>
double CVFunction<CVType, MLAlgorithm, TotalArgs, BoundArgs...>::Evaluate(
    const arma::mat& parameters,
    const Args&... args)
{
  return PutNextArg<BoundArgIndex, ParamIndex>(parameters, args...);
}

template<typename CVType,
         typename MLAlgorithm,
         size_t TotalArgs,
         typename... BoundArgs>
template<size_t BoundArgIndex,
         size_t ParamIndex,
         typename... Args,
         typename,
         typename>
double CVFunction<CVType, MLAlgorithm, TotalArgs, BoundArgs...>::Evaluate(
    const arma::mat& /* parameters */,
    const Args&... args)
{
  double objective = cv.Evaluate(args...);

  // Change the best model if we have got a better score, or if we probably
  // have not assigned any valid (trained) model yet.
  if (bestObjective > objective ||
      bestObjective == std::numeric_limits<double>::max())
  {
    bestObjective = objective;
    bestModel = std::move(cv.Model());
  }

  return objective;
}

template<typename CVType,
         typename MLAlgorithm,
         size_t TotalArgs,
         typename... BoundArgs>
template<size_t BoundArgIndex,
         size_t ParamIndex,
         typename... Args,
         typename>
double CVFunction<CVType, MLAlgorithm, TotalArgs, BoundArgs...>::PutNextArg(
    const arma::mat& parameters,
    const Args&... args)
{
  return Evaluate<BoundArgIndex + 1, ParamIndex>(
      parameters, args..., std::get<BoundArgIndex>(boundArgs).value);
}

template<typename CVType,
         typename MLAlgorithm,
         size_t TotalArgs,
         typename... BoundArgs>
template<size_t BoundArgIndex,
         size_t ParamIndex,
         typename... Args,
         typename,
         typename>
double CVFunction<CVType, MLAlgorithm, TotalArgs, BoundArgs...>::PutNextArg(
    const arma::mat& parameters,
    const Args&... args)
{
  if (datasetInfo.Type(ParamIndex) == data::Datatype::categorical)
  {
    return Evaluate<BoundArgIndex, ParamIndex + 1>(parameters, args...,
        datasetInfo.UnmapString(size_t(parameters(ParamIndex, 0)), ParamIndex));
  }
  else
  {
    return Evaluate<BoundArgIndex, ParamIndex + 1>(parameters, args...,
        parameters(ParamIndex, 0));
  }
}

} // namespace mlpack

#endif
