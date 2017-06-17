/**
 * @file gradient_update_visitor.hpp
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
#ifndef MLPACK_METHODS_ANN_VISITOR_GRADIENT_UPDATE_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_GRADIENT_UPDATE_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * GradientUpdateVisitor update the gradient parameter given the gradient set.
 */
class GradientUpdateVisitor : public boost::static_visitor<size_t>
{
 public:
  //! Update the gradient parameter given the gradient set.
  GradientUpdateVisitor(arma::mat&& gradient, size_t offset = 0);

  //! Update the gradient parameter.
  template<typename LayerType>
  size_t operator()(LayerType* layer) const;

 private:
  //! The gradient set.
  arma::mat&& gradient;

  //! The gradient offset.
  size_t offset;

  //! Update the gradient if the module implements the Gradient() function.
  template<typename T>
  typename std::enable_if<
      HasGradientCheck<T, arma::mat&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Update the gradient if the module implements the Model() function.
  template<typename T>
  typename std::enable_if<
      !HasGradientCheck<T, arma::mat&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Update the gradient if the module implements the Gradient() and Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasGradientCheck<T, arma::mat&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerGradients(T* layer, arma::mat& input) const;

  //! Do not update the gradient parameter if the module doesn't implement the
  //! Gradient() or Model() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasGradientCheck<T, P&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
  LayerGradients(T* layer, P& input) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "gradient_update_visitor_impl.hpp"

#endif
