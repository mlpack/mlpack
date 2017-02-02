/**
 * @file weight_set_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Weight() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_WEIGHT_SET_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_WEIGHT_SET_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "weight_set_visitor.hpp"

namespace mlpack {
namespace ann {

//! WeightSetVisitor visitor class.
inline WeightSetVisitor::WeightSetVisitor(arma::mat&& weight,
                                          const size_t offset) :
    weight(std::move(weight)),
    offset(offset)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline size_t WeightSetVisitor::operator()(LayerType* layer) const
{
  return LayerSize(layer, std::move(layer->OutputParameter()));
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasParametersCheck<T, P&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSetVisitor::LayerSize(T* /* layer */, P&& /*output */) const
{
  return 0;
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasParametersCheck<T, P&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSetVisitor::LayerSize(T* layer, P&& /*output */) const
{
  size_t modelOffset = 0;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    modelOffset += boost::apply_visitor(WeightSetVisitor(
        std::move(weight), modelOffset + offset), layer->Model()[i]);
  }

  return modelOffset;
}

template<typename T, typename P>
inline typename std::enable_if<
    HasParametersCheck<T, P&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSetVisitor::LayerSize(T* layer, P&& /* output */) const
{
  layer->Parameters() = arma::mat(weight.memptr() + offset,
      layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);

  return layer->Parameters().n_elem;
}

template<typename T, typename P>
inline typename std::enable_if<
    HasParametersCheck<T, P&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSetVisitor::LayerSize(T* layer, P&& /* output */) const
{
  layer->Parameters() = arma::mat(weight.memptr() + offset,
      layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);

  size_t modelOffset = layer->Parameters().n_elem;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    modelOffset += boost::apply_visitor(WeightSetVisitor(
        std::move(weight), modelOffset + offset), layer->Model()[i]);
  }

  return modelOffset;
}

} // namespace ann
} // namespace mlpack

#endif
