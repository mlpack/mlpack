/**
 * @file gradient_zero_visitor.hpp
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
#ifndef MLPACK_METHODS_ANN_VISITOR_GRADIENT_ZERO_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_GRADIENT_ZERO_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/*
 * GradientZeroVisitor set the gradient to zero for the given module.
 */
class GradientZeroVisitor : public boost::static_visitor<void>
{
 public:
  //! Set the gradient to zero for the given module.
  GradientZeroVisitor();

  //! Set the gradient to zero.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! Set the gradient to zero if the module implements the Gradient() function.
  template<typename T>
  typename std::enable_if<
      HasGradientCheck<T, arma::mat&(T::*)()>::value, void>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Do not set the gradient to zero if the module doesn't implement the
  //! Gradient() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasGradientCheck<T, P&(T::*)()>::value, void>::type
  LayerGradients(T* layer, P& input) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "gradient_zero_visitor_impl.hpp"

#endif
