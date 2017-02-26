/**
 * @file gradient_set_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Gradient() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_GRADIENT_SET_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_GRADIENT_SET_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "gradient_set_visitor.hpp"

namespace mlpack {
namespace ann {

//! GradientSetVisitor visitor class.
inline GradientSetVisitor::GradientSetVisitor(arma::mat&& gradient,
                                              size_t offset) :
    gradient(std::move(gradient)),
    offset(offset)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline size_t GradientSetVisitor::operator()(LayerType* layer) const
{
  return LayerGradients(layer, layer->OutputParameter());
}

template<typename T>
inline typename std::enable_if<
    HasGradientCheck<T, arma::mat&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
GradientSetVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
{
  layer->Gradient() = arma::mat(gradient.memptr() + offset,
      layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);

  return layer->Parameters().n_elem;
}

template<typename T>
inline typename std::enable_if<
    !HasGradientCheck<T, arma::mat&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
GradientSetVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
{
  size_t modelOffset = 0;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    modelOffset += boost::apply_visitor(GradientSetVisitor(
        std::move(gradient), modelOffset + offset), layer->Model()[i]);
  }

  return modelOffset;
}

template<typename T>
inline typename std::enable_if<
    HasGradientCheck<T, arma::mat&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
GradientSetVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
{
  layer->Gradient() = arma::mat(gradient.memptr() + offset,
      layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);

  size_t modelOffset = layer->Parameters().n_elem;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    modelOffset += boost::apply_visitor(GradientSetVisitor(
        std::move(gradient), modelOffset + offset), layer->Model()[i]);
  }

  return modelOffset;
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasGradientCheck<T, P&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
GradientSetVisitor::LayerGradients(T* /* layer */, P& /* input */) const
{
  return 0;
}

} // namespace ann
} // namespace mlpack

#endif
