/**
 * @file weight_size_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the WeightSize() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_WEIGHT_SIZE_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_WEIGHT_SIZE_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "weight_size_visitor.hpp"

namespace mlpack {
namespace ann {

//! WeightSizeVisitor visitor class.
template<typename LayerType>
inline size_t WeightSizeVisitor::operator()(LayerType* layer) const
{
  return LayerSize(layer, layer->OutputParameter());
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasParametersCheck<T, P&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSizeVisitor::LayerSize(T* /* layer */, P& /* output */) const
{
  return 0;
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasParametersCheck<T, P&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSizeVisitor::LayerSize(T* layer, P& /* output */) const
{
  size_t weights = 0;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    weights += boost::apply_visitor(WeightSizeVisitor(), layer->Model()[i]);
  }

  return weights;
}

template<typename T, typename P>
inline typename std::enable_if<
    HasParametersCheck<T, P&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSizeVisitor::LayerSize(T* layer, P& /* output */) const
{
  return layer->Parameters().n_elem;
}

template<typename T, typename P>
inline typename std::enable_if<
    HasParametersCheck<T, P&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSizeVisitor::LayerSize(T* layer, P& /* output */) const
{
  size_t weights = layer->Parameters().n_elem;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    weights += boost::apply_visitor(WeightSizeVisitor(), layer->Model()[i]);
  }

  return weights;
}

} // namespace ann
} // namespace mlpack

#endif
