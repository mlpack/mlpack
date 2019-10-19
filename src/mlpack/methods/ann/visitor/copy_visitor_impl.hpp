/**
 * @file copy_visitor_impl.hpp
 * @author Shangtong Zhang
 *
 * This file provides an implementation for copy between layers
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_COPY_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_COPY_VISITOR_IMPL_HPP

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

template <typename... CustomLayers>
template <typename LayerType>
inline LayerTypes<CustomLayers...>
CopyVisitor<CustomLayers...>::operator()(LayerType* layer) const
{
  return new LayerType(*layer);
}

template <typename... CustomLayers>
inline LayerTypes<CustomLayers...>
CopyVisitor<CustomLayers...>::operator()(MoreTypes layer) const
{
  return layer.apply_visitor(*this);
}

} // namespace ann
} // namespace mlpack

#endif
