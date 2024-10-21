/**
 * @file core/hpt/hpt.hpp
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
#include <mlpack/core/hpt/cv_function.hpp>
#include <ensmallen.hpp>

namespace mlpack {

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
 * for them with the Fixed function as in the following example of finding good
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
 * std::tie(bestLambda1, bestLambda2) = hpt2.Optimize(Fixed(transposeData),
 *     Fixed(useCholesky), lambda1Set, lambda2Set);
 * @endcode
 *
 * @tparam MLAlgorithm A machine learning algorithm.
 * @tparam Metric A metric to assess the quality of a trained model.
 * @tparam CV A cross-validation strategy used to assess a set of
 *     hyper-parameters.
 * @tparam OptimizerType An optimization strategy (GridSearch and
 *     GradientDescent are supported).
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
         typename OptimizerType = ens::GridSearch,
         typename MatType = arma::mat,
         typename PredictionsType =
             typename MetaInfoExtractor<MLAlgorithm, MatType>::PredictionsType,
         typename WeightsType =
             typename MetaInfoExtractor<MLAlgorithm, MatType,
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

  //! Access and modify the optimizer.
  OptimizerType& Optimizer() { return optimizer; }

  /**
   * Get relative increase of arguments for calculation of partial
   * derivatives (by the definition) in gradient-based optimization. The exact
   * increase for some particular argument is equal to the absolute value of the
   * argument multiplied by the relative increase (see also the documentation
   * for MinDelta()).
   *
   * The default value is 0.01.
   */
  double RelativeDelta() const { return relativeDelta; }

  /**
   * Modify relative increase of arguments for calculation of partial
   * derivatives (by the definition) in gradient-based optimization. The exact
   * increase for some particular argument is equal to the absolute value of the
   * argument multiplied by the relative increase (see also the documentation
   * for MinDelta()).
   *
   * The default value is 0.01.
   */
  double& RelativeDelta() { return relativeDelta; }

  /**
   * Get minimum increase of arguments for calculation of partial derivatives
   * (by the definition) in gradient-based optimization. This value is going to
   * be used when it is greater than the increase calculated with the rules
   * described in the documentation for RelativeDelta().
   *
   * The default value is 1e-10.
   */
  double MinDelta() const { return minDelta; }

  /**
   * Modify minimum increase of arguments for calculation of partial derivatives
   * (by the definition) in gradient-based optimization. This value is going to
   * be used when it is greater than the increase calculated with the rules
   * described in the documentation for RelativeDelta().
   *
   * The default value is 1e-10.
   */
  double& MinDelta() { return minDelta; }

  /**
   * Find the best hyper-parameters by using the given Optimizer. For each
   * hyper-parameter one of the following should be passed as an argument.
   * 1. A set of values to choose from (when using GridSearch as an optimizer).
   *   The set of values should be an STL-compatible container (it should
   *   provide begin() and end() methods returning iterators).
   * 2. A starting value (when using any other optimizer than GridSearch).
   * 3. A value fixed by using the function `Fixed`. In this case the
   *   hyper-parameter will not be optimized.
   *
   * All arguments should be passed in the same order as if the corresponding
   * hyper-parameters would be passed into the Evaluate method of the given CV
   * class (in the order as they appear in the constructor(s) of the given
   * MLAlgorithm). Also, arguments for all required hyper-parameters (ones that
   * don't have default values in the corresponding MLAlgorithm constructor)
   * should be provided.
   *
   * The method returns a tuple of values for hyper-parameters that haven't been
   * fixed.
   *
   * @param args Arguments corresponding to hyper-parameters (see the method
   *   description for more information).
   */
  template<typename... Args>
  TupleOfHyperParameters<Args...> Optimize(const Args&... args);

  //! Get the performance measurement of the best model from the last run.
  double BestObjective() const { return bestObjective; }

  //! Get the best model from the last run.
  const MLAlgorithm& BestModel() const { return bestModel; }

  //! Modify the best model from the last run.
  MLAlgorithm& BestModel() { return bestModel; }

 private:
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
  using CVType = std::conditional_t<Metric::NeedsMinimization,
      CV<MLAlgorithm, Metric, MatType, PredictionsType, WeightsType>,
      CV<MLAlgorithm, Negated<Metric>, MatType, PredictionsType, WeightsType>>;


  //! The cross-validation object for assessing sets of hyper-parameters.
  CVType cv;

  //! The optimizer.
  OptimizerType optimizer;

  //! The best objective from the last run.
  double bestObjective;

  //! The best model from the last run.
  MLAlgorithm bestModel;

  /**
   * The relative increase of arguments for calculation of gradient in
   * CVFunction.
   */
  double relativeDelta;

  /**
   * The minimum increase of arguments for calculation of gradient in
   * CVFunction.
   */
  double minDelta;

  /**
   * A type function to check whether the element I of the tuple type is a
   * PreFixedArg.
   */
  template<typename Tuple, size_t I>
  using IsPreFixed = IsPreFixedArg<std::tuple_element_t<I, Tuple>>;

  /**
   * A type function to check whether the element I of the tuple type is an
   * arithmetic type.
   */
  template<typename Tuple, size_t I>
  using IsArithmetic = std::is_arithmetic<std::remove_reference_t<
      std::tuple_element_t<I, Tuple>>>;

  /**
   * The set of methods to initialize auxiliary objects (a CVFunction object and
   * the datasetInfo parameter) and run optimization to find the best
   * hyper-parameters.
   *
   * This template is called when we are ready to run optimization.
   */
  template<size_t I /* Index of the next argument to handle. */,
           typename ArgsTuple,
           typename... FixedArgs,
           typename = std::enable_if_t<I == std::tuple_size<ArgsTuple>::value>>
  inline void InitAndOptimize(
      const ArgsTuple& args,
      arma::mat& bestParams,
      data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
      FixedArgs... fixedArgs);

  /**
   * The set of methods to initialize auxiliary objects (a CVFunction object and
   * the datasetInfo parameter) and run optimization to find the best
   * hyper-parameters.
   *
   * This template is called when the next argument should be fixed (should not
   * be optimized).
   */
  template<size_t I /* Index of the next argument to handle. */,
           typename ArgsTuple,
           typename... FixedArgs,
           typename = std::enable_if_t<(I < std::tuple_size<ArgsTuple>::value)>,
           typename = std::enable_if_t<IsPreFixed<ArgsTuple, I>::value>>
  inline void InitAndOptimize(
      const ArgsTuple& args,
      arma::mat& bestParams,
      data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
      FixedArgs... fixedArgs);

  /**
   * The set of methods to initialize auxiliary objects (a CVFunction object and
   * the datasetInfo parameter) and run optimization to find the best
   * hyper-parameters.
   *
   * This template is called when the next argument is of an arithmetic type and
   * should be used as an initial value for the hyper-parameter.
   */
  template<size_t I /* Index of the next argument to handle. */,
           typename ArgsTuple,
           typename... FixedArgs,
           typename = std::enable_if_t<(I < std::tuple_size<ArgsTuple>::value)>,
           typename = std::enable_if_t<!IsPreFixed<ArgsTuple, I>::value &&
                   IsArithmetic<ArgsTuple, I>::value>,
           typename = void>
  inline void InitAndOptimize(
      const ArgsTuple& args,
      arma::mat& bestParams,
      data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
      FixedArgs... fixedArgs);

  /**
   * The set of methods to initialize auxiliary objects (a CVFunction object and
   * the datasetInfo parameter) and run optimization to find the best
   * hyper-parameters.
   *
   * This template is called when the next argument should be used to specify
   * possible values for the hyper-parameter in datasetInfo.
   */
  template<size_t I /* Index of the next argument to handle. */,
           typename ArgsTuple,
           typename... FixedArgs,
           typename = std::enable_if_t<(I < std::tuple_size<ArgsTuple>::value)>,
           typename = std::enable_if_t<!IsPreFixed<ArgsTuple, I>::value &&
                   !IsArithmetic<ArgsTuple, I>::value>,
           typename = void,
           typename = void>
  inline void InitAndOptimize(
      const ArgsTuple& args,
      arma::mat& bestParams,
      data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
      FixedArgs... fixedArgs);

  /**
   * Gather all elements of vector in an argument list and use them to create a
   * tuple.
   */
  template<typename TupleType,
           size_t I /* Index of the element in vector to handle. */,
           typename... Args,
           typename = typename
               std::enable_if_t<(I < std::tuple_size<TupleType>::value)>>
  inline TupleType VectorToTuple(const arma::vec& vector, const Args&... args);

  /**
   * Create a tuple from args.
   */
  template<typename TupleType,
           size_t I /* Index of the element in vector to handle. */,
           typename... Args,
           typename = typename
               std::enable_if_t<I == std::tuple_size<TupleType>::value>,
           typename = void>
  inline TupleType VectorToTuple(const arma::vec& vector, const Args&... args);
};

} // namespace mlpack

// Include implementation
#include "hpt_impl.hpp"

#endif
