/**
 * @file core/hpt/cv_function.hpp
 * @author Kirill Mishchenko
 *
 * A cross-validation wrapper for optimizers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_HPT_CV_FUNCTION_HPP
#define MLPACK_CORE_HPT_CV_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * This wrapper serves for adapting the interface of the cross-validation
 * classes to the one that can be utilized by the mlpack optimizers.
 *
 * This class is not supposed to be used directly by users. To tune
 * hyper-parameters see HyperParameterTuner.
 *
 * @tparam CVType A cross-validation strategy.
 * @tparam MLAlgorithm The machine learning algorithm used in cross-validation.
 * @tparam TotalArgs The total number of arguments that are supposed to be
 *     passed to the Evaluate method of a CVType object.
 * @tparam BoundArgs Types of arguments (wrapped into the BoundArg struct) that
 *     should be passed into the Evaluate method of a CVType object but are not
 *     going to be passed into the Evaluate method of a CVFunction object.
 */
template<typename CVType,
         typename MLAlgorithm,
         size_t TotalArgs,
         typename... BoundArgs>
class CVFunction
{
 public:
  /**
   * Initialize a CVFunction object.
   *
   * @param cv A cross-validation object.
   * @param datasetInfo Information on each parameter (categorical/numeric).
   *     Contains mappings from optimizer-passed size_t indices to double values
   *     that should be used.
   * @param relativeDelta Relative increase of arguments for calculation of
   *     partial derivatives (by the definition). The exact increase for some
   *     particular argument is equal to the absolute value of the argument
   *     multiplied by the relative increase (see also the documentation for the
   *     minDelta parameter).
   * @param minDelta Minimum increase of arguments for calculation of partial
   *     derivatives (by the definition). This value is going to be used when it
   *     is greater than the increase calculated with the rules described in the
   *     documentation for the relativeDelta parameter.
   * @param args Arguments that should be passed into the Evaluate method
   *     of the CVType object but are not going to be passed into the Evaluate
   *     method of this object.
   */
  CVFunction(CVType& cv,
             data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
             const double relativeDelta,
             const double minDelta,
             const BoundArgs&... args);

  /**
   * Run cross-validation with the bound and passed parameters.
   *
   * @param parameters Arguments (rather than the bound arguments) that should
   *     be passed into the Evaluate method of the CVType object.
   */
  double Evaluate(const arma::mat& parameters);

  /**
   * Evaluate numerically the gradient of the CVFunction with the given
   * parameters.
   *
   * @param parameters Arguments (rather than the bound arguments) that should
   *     be passed into the Evaluate method of the CVType object.
   * @param gradient Vector to output the gradient into.
   */
  void Gradient(const arma::mat& parameters, arma::mat& gradient);

  //! Access and modify the best model so far.
  MLAlgorithm& BestModel() { return bestModel; }

 private:
  //! The type of tuples of BoundArgs.
  using BoundArgsTupleType = std::tuple<BoundArgs...>;

  //! The amount of bound arguments.
  static const size_t BoundArgsAmount =
      std::tuple_size<BoundArgsTupleType>::value;

  /**
   * A struct that finds out whether the next argument for the Evaluate method
   * of a CVType object should be a bound argument at the position BoundArgIndex
   * rather than an element of parameters at the position ParamIndex.
   */
  template<size_t BoundArgIndex,
           size_t ParamIndex,
           bool BoundArgsIndexInRange = (BoundArgIndex < BoundArgsAmount)>
  struct UseBoundArg;

  //! A reference to the cross-validation object.
  CVType& cv;

  //! Information on each argument to be optimized.
  data::DatasetMapper<data::IncrementPolicy, double> datasetInfo;

  //! The bound arguments.
  BoundArgsTupleType boundArgs;

  //! The best objective so far.
  double bestObjective;

  //! The best model so far.
  MLAlgorithm bestModel;

  //! Relative increase of arguments for calculation of gradient.
  double relativeDelta;

  //! Minimum absolute increase of arguments for calculation of gradient.
  double minDelta;

  /**
   * Collect all arguments and run cross-validation.
   */
  template<size_t BoundArgIndex,
           size_t ParamIndex,
           typename... Args,
           typename =
               std::enable_if_t<(BoundArgIndex + ParamIndex < TotalArgs)>>
  inline double Evaluate(const arma::mat& parameters, const Args&... args);

  /**
   * Run cross-validation with the collected arguments.
   */
  template<size_t BoundArgIndex,
           size_t ParamIndex,
           typename... Args,
           typename =
               std::enable_if_t<BoundArgIndex + ParamIndex == TotalArgs>,
           typename = void>
  inline double Evaluate(const arma::mat& parameters, const Args&... args);

  /**
   * Put the bound argument (at the BoundArgIndex position) as the next one.
   */
  template<size_t BoundArgIndex,
           size_t ParamIndex,
           typename... Args,
           typename = std::enable_if_t<
               UseBoundArg<BoundArgIndex, ParamIndex>::value>>
  inline double PutNextArg(const arma::mat& parameters, const Args&... args);

  /**
   * Put the element (at the ParamIndex position) of the parameters as the next
   * one.
   */
  template<size_t BoundArgIndex,
           size_t ParamIndex,
           typename... Args,
           typename = std::enable_if_t<
               !UseBoundArg<BoundArgIndex, ParamIndex>::value>,
           typename = void>
  inline double PutNextArg(const arma::mat& parameters, const Args&... args);
};


} // namespace mlpack

// Include implementation
#include "cv_function_impl.hpp"

#endif
