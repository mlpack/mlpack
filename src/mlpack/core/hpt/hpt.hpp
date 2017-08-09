/**
 * @file hpt.hpp
 * @author Kirill Mishchenko
 *
 * Hyper-parameter tuning.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_HPT_HPT_HPP
#define MLPACK_CORE_HPT_HPT_HPP

#include <mlpack/core/cv/meta_info_extractor.hpp>
#include <mlpack/core/hpt/deduce_hp_types.hpp>
#include <mlpack/core/optimizers/grid_search/grid_search.hpp>

namespace mlpack {
namespace hpt {

/**
 * The class HyperParameterTuner for the given MLAlgorithm utilizes the provided
 * Optimizer to find the values of hyper-parameters that optimize the value of
 * the given Metric. The value of the Metric is calculated by performing
 * cross-validation with the provided cross-validation strategy.
 *
 * To construct a HyperParameterTuner object you need to pass the same arguments
 * as for construction of an object of the given CV class. For example, we can
 * use the following code to try to find a good lambda value for
 * LinearRegression.
 *
 * @code
 * // 100-point 5-dimensional random dataset.
 * arma::mat data = arma::randu<arma::mat>(5, 100);
 * // Noisy responses retrieved by a random linear transformation of data.
 * arma::rowvec responses = arma::randu<arma::rowvec>(5) * data +
 *     0.1 * arma::randn<arma::rowvec>(100);
 *
 * // Using 80% of data for training and remaining 20% for assessing MSE.
 * double validationSize = 0.2;
 * HyperParameterTuner<LinearRegression, MSE, SimpleCV> hpt(validationSize,
 *     data, responses);
 *
 * // Finding the best value for lambda from the values 0.0, 0.001, 0.01, 0.1,
 * // and 1.0.
 * arma::vec lambdas{0.0, 0.001, 0.01, 0.1, 1.0};
 * double bestLambda;
 * std::tie(bestLambda) = hpt.Optimize(lambdas);
 * @endcode
 *
 * When some hyper-parameters should not be optimized, you can specify values
 * for them with the Bind function as in the following example of finding good
 * lambda1 and lambda2 values for LARS.
 *
 * @code
 * HyperParameterTuner<LARS, MSE, SimpleCV> hpt2(validationSize, data,
 *     responses);
 *
 * bool transposeData = true;
 * bool useCholesky = false;
 * arma::vec lambda1Set{0.0, 0.001, 0.01, 0.1, 1.0};
 * arma::vec lambda2Set{0.0, 0.002, 0.02, 0.2, 2.0};
 *
 * double bestLambda1, bestLambda2;
 * std::tie(bestLambda1, bestLambda2) = hpt2.Optimize(Bind(transposeData),
 *     Bind(useCholesky), lambda1Set, lambda2Set);
 * @endcode
 *
 * @tparam MLAlgorithm A machine learning algorithm.
 * @tparam Metric A metric to assess the quality of a trained model.
 * @tparam CV A cross-validation strategy used to assess a set of
 *     hyper-parameters.
 * @tparam Optimizer An optimization strategy.
 * @tparam MatType The type of data.
 * @tparam PredictionsType The type of predictions (should be passed when the
 *     predictions type is a template parameter in Train methods of the given
 *     MLAlgorithm; arma::Row<size_t> will be used otherwise).
 * @tparam WeightsType The type of weights (should be passed when weighted
 *     learning is supported, and the weights type is a template parameter in
 *     Train methods of the given MLAlgorithm; arma::vec will be used
 *     otherwise).
 */
template<typename MLAlgorithm,
         typename Metric,
         template<typename, typename, typename, typename, typename> class CV,
         typename Optimizer = mlpack::optimization::GridSearch,
         typename MatType = arma::mat,
         typename PredictionsType =
             typename cv::MetaInfoExtractor<MLAlgorithm,
                 MatType>::PredictionsType,
         typename WeightsType =
             typename cv::MetaInfoExtractor<MLAlgorithm, MatType,
                 PredictionsType>::WeightsType>
class HyperParameterTuner
{
 public:
  /**
   * Create a HyperParameterTuner object by passing constructor arguments for
   * the given cross-validation strategy (the CV class).
   *
   * @param args Constructor arguments for the given cross-validation
   *     strategy (the CV class).
   */
  template<typename... CVArgs>
  HyperParameterTuner(const CVArgs& ...args);

  /**
   * Find the best hyper-parameters by using the given Optimizer. For each
   * hyper-parameter one of the following should be passed as an argument.
   * 1. A set of values to choose from (when using GridSearch as an optimizer).
   *   The set of values should be an STL-compatible container (it should
   *   provide begin() and end() methods returning iterators).
   * 2. A starting value (when using any other optimizer than GridSearch).
   * 3. A value bound by using the function mlpack::hpt::Bind. In this case the
   *   hyper-parameter will not be optimized.
   *
   * All arguments should be passed in the same order as if the corresponding
   * hyper-paramters would be passed into the Evaluate method of the given CV
   * class (in the order as they appear in the constructor(s) of the given
   * MLAlgorithm). Also, arguments for all required hyper-parameters (ones that
   * don't have default values in the corresponding MLAlgorithm constructor)
   * should be provided.
   *
   * The method returns a tuple of values for hyper-parameters that haven't been
   * bound.
   *
   * @param args Arguments corresponding to hyper-parameters (see the method
   *   description for more information).
   */
  template<typename... Args>
  TupleOfHyperParameters<Args...> Optimize(const Args&... args);

  //! Access the performance measurement of the best model from the last run.
  double BestObjective() const { return bestObjective; }

  //! Access and modify the best model from the last run.
  MLAlgorithm& BestModel() { return bestModel; }

 private:
  static_assert(
      std::is_same<Optimizer, mlpack::optimization::GridSearch>::value,
      "HyperParameterTuner now supports only the GridSearch optimizer");

  /**
   * A decorator that returns negated values of the original metric.
   */
  template<typename OriginalMetric>
  struct Negated
  {
    static double Evaluate(MLAlgorithm& model,
                           const MatType& xs,
                           const PredictionsType& ys)
    { return -OriginalMetric::Evaluate(model, xs, ys); }
  };

  //! A short alias for the full type of the cross-validation.
  using CVType = typename std::conditional<Metric::NeedsMinimization,
      CV<MLAlgorithm, Metric, MatType, PredictionsType, WeightsType>,
      CV<MLAlgorithm, Negated<Metric>, MatType, PredictionsType,
          WeightsType>>::type;


  //! The cross-validation object for assessing sets of hyper-parameters.
  CVType cv;

  //! The optimizer.
  Optimizer optimizer;

  //! The best objective from the last run.
  double bestObjective;

  //! The best model from the last run.
  MLAlgorithm bestModel;

  /**
   * The set of methods to initialize the GridSearch optimizer. We basically
   * need to go through all arguments passed to the Optimize method, gather all
   * non-bound arguments (collections) and pass them into the GridSearch
   * constructor.
   */
  template<size_t I,
           typename ArgsTuple,
           typename... Collections,
           typename = std::enable_if_t<I == std::tuple_size<ArgsTuple>::value>>
  inline void InitGridSearch(const ArgsTuple& args,
                             Collections... collections);

  template<size_t I,
           typename ArgsTuple,
           typename... Collections,
           typename = std::enable_if_t<I < std::tuple_size<ArgsTuple>::value>,
           typename = std::enable_if_t<IsPreBoundArg<
               typename std::tuple_element<I, ArgsTuple>::type>::value>>
  inline void InitGridSearch(const ArgsTuple& args,
                             Collections... collections);

  template<size_t I,
           typename ArgsTuple,
           typename... Collections,
           typename = std::enable_if_t<I < std::tuple_size<ArgsTuple>::value>,
           typename = std::enable_if_t<!IsPreBoundArg<
               typename std::tuple_element<I, ArgsTuple>::type>::value>,
           typename = void>
  inline void InitGridSearch(const ArgsTuple& args,
                             Collections... collections);

  /**
   * The set of methods to initialize a cost function (CVFunction) object and
   * run optimization to find the best hyper-parameters. To initialize a
   * CVFunction object we go through all arguments passed to the Optimize
   * method, gather all bound values and pass them into the CVFunction
   * constructor.
   */
  template<size_t I,
           typename ArgsTuple,
           typename... BoundArgs,
           typename = std::enable_if_t<I == std::tuple_size<ArgsTuple>::value>>
  inline void InitCVFunctionAndOptimize(const ArgsTuple& args,
                                        arma::mat& bestParams,
                                        BoundArgs... boundArgs);

  template<size_t I,
           typename ArgsTuple,
           typename... BoundArgs,
           typename = std::enable_if_t<I < std::tuple_size<ArgsTuple>::value>,
           typename = std::enable_if_t<IsPreBoundArg<
               typename std::tuple_element<I, ArgsTuple>::type>::value>>
  inline void InitCVFunctionAndOptimize(const ArgsTuple& args,
                                        arma::mat& bestParams,
                                        BoundArgs... boundArgs);

  template<size_t I,
           typename ArgsTuple,
           typename... BoundArgs,
           typename = std::enable_if_t<I < std::tuple_size<ArgsTuple>::value>,
           typename = std::enable_if_t<!IsPreBoundArg<
               typename std::tuple_element<I, ArgsTuple>::type>::value>,
           typename = void>
  inline void InitCVFunctionAndOptimize(const ArgsTuple& args,
                                        arma::mat& bestParams,
                                        BoundArgs... boundArgs);

  /**
   * The set of methods to convert the given arma::vec vector to the tuple of
   * the target type by gathering all elements in an argument list.
   */
  template<typename TupleType,
           size_t I,
           typename... Args,
           typename = typename
               std::enable_if_t<I < std::tuple_size<TupleType>::value>>
  inline TupleType VectorToTuple(const arma::vec& vector, const Args&... args);

  template<typename TupleType,
           size_t I,
           typename... Args,
           typename = typename
               std::enable_if_t<I == std::tuple_size<TupleType>::value>,
           typename = void>
  inline TupleType VectorToTuple(const arma::vec& vector, const Args&... args);
};

} // namespace hpt
} // namespace mlpack

// Include implementation
#include "hpt_impl.hpp"

#endif
