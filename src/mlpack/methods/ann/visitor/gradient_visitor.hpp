/**
 * @file methods/ann/visitor/gradient_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the Gradient() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_GRADIENT_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_GRADIENT_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * SearchModeVisitor executes the Gradient() method of the given module using
 * the input and delta parameter.
 */
class GradientVisitor : public boost::static_visitor<void>
{
 public:
  //! Executes the Gradient() method of the given module using the input and
  //! delta parameter.
  GradientVisitor(const arma::mat& input, const arma::mat& delta);

  //! Executes the Gradient() method for the layer with the specified index.
  GradientVisitor(const arma::mat& input,
                  const arma::mat& delta,
                  const size_t index);

  //! Executes the Gradient() method.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

  void operator()(MoreTypes layer) const;

 private:
  //! The input set.
  const arma::mat& input;

  //! The delta parameter.
  const arma::mat& delta;

  //! Index of the layer to run.
  size_t index;

  //! Indicates whether to use index or not
  bool hasIndex;

  //! Execute the Gradient() function if the module implements the Gradient()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasGradientCheck<T, arma::mat&(T::*)()>::value &&
      !HasRunCheck<T, bool&(T::*)(void)>::value, void>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Execute the Gradient() function if the module implements the Gradient()
  //! and has a Run() function.
  template<typename T>
  typename std::enable_if<
      HasGradientCheck<T, arma::mat&(T::*)()>::value &&
      HasRunCheck<T, bool&(T::*)(void)>::value, void>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Do not execute the Gradient() function if the module doesn't implement
  //! the Gradient() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasGradientCheck<T, P&(T::*)()>::value, void>::type
  LayerGradients(T* layer, P& input) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "gradient_visitor_impl.hpp"

#endif
