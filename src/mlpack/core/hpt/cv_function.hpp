/**
 * @file cv_function.hpp
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
namespace hpt {

/**
 * This wrapper serves for adapting the interface of the cross-validation
 * classes to the one that can be utilized by the mlpack optimizers.
 *
 * This class is not supposed to be used directly by users. To tune
 * hyper-parameters see HyperParameterTuner.
 *
 * @tparam CVType A cross-validation strategy.
 * @tparam TotalArgs The total number of arguments that are supposed to be
 *     passed to the Evaluate method of a CVType object.
 * @tparam BoundArgs Types of arguments (wrapped into the BoundArg struct) that
 *     should be passed into the Evaluate method of a CVType object but are not
 *     going to be passed into the Evaluate method of a CVFunction object.
 */
template<typename CVType, size_t TotalArgs, typename... BoundArgs>
class CVFunction
{
 public:
  /**
   * Initialize a CVFunction object.
   *
   * @param cv A cross-validation object.
   * @param BoundArgs Arguments that should be passed into the Evaluate method
   *     of the CVType object but are not going to be passed into the Evaluate
   *     method of this object.
   */
  CVFunction(CVType& cv, const BoundArgs&... args);

  /**
   * Run cross-validation with the bound and passed parameters.
   *
   * @param parameters Arguments (rather than the bound arguments) that should
   *     be passed into the Evaluate method of the CVType object.
   */
  double Evaluate(const arma::mat& parameters);

  //! The used machine learning algorithm.
  using MLAlgorithm = typename
      std::remove_reference<decltype(std::declval<CVType>().Model())>::type;

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
   * of a CVType object should be a bound argument at the position BAIndex
   * rather than an element of parameters at the position PIndex.
   */
  template<size_t BAIndex,
           size_t PIndex,
           bool BoundArgsIndexInRange = BAIndex < BoundArgsAmount>
  struct UseBoundArg;

  //! A reference to the cross-validation object.
  CVType& cv;

  //! The bound arguments.
  BoundArgsTupleType boundArgs;

  //! The best objective so far.
  double bestObjective;

  //! The best model so far.
  MLAlgorithm bestModel;

  /**
   * Collect all arguments and run cross-validation.
   */
  template<size_t BAIndex,
           size_t PIndex,
           typename... Args,
           typename =
               typename std::enable_if<BAIndex + PIndex < TotalArgs>::type>
  inline double Evaluate(const arma::mat& parameters, const Args&... args);

  /**
   * Run cross-validation with the collected arguments.
   */
  template<size_t BAIndex,
           size_t PIndex,
           typename... Args,
           typename =
               typename std::enable_if<BAIndex + PIndex == TotalArgs>::type,
           typename = void>
  inline double Evaluate(const arma::mat& parameters, const Args&... args);

  /**
   * Put the bound argument (at the BAIndex position) as the next one.
   */
  template<size_t BAIndex,
           size_t PIndex,
           typename... Args,
           typename = typename std::enable_if<
               UseBoundArg<BAIndex, PIndex>::value>::type>
  inline double PutNextArg(const arma::mat& parameters, const Args&... args);

  /**
   * Put the element (at the PIndex position) of the parameters as the next one.
   */
  template<size_t BAIndex,
           size_t PIndex,
           typename... Args,
           typename = typename std::enable_if<
               !UseBoundArg<BAIndex, PIndex>::value>::type,
           typename = void>
  inline double PutNextArg(const arma::mat& parameters, const Args&... args);
};


} // namespace hpt
} // namespace mlpack

// Include implementation
#include "cv_function_impl.hpp"

#endif
