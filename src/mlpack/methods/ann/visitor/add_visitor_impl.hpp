/**
 * @file methods/ann/visitor/add_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Add() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_ADD_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_ADD_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "add_visitor.hpp"

namespace mlpack {
namespace ann {

//! AddVisitor visitor class.
template<typename... CustomLayers>
template<typename T>
inline AddVisitor<CustomLayers...>::AddVisitor(T newLayer) :
    newLayer(std::move(newLayer))
{
  /* Nothing to do here. */
}

template<typename... CustomLayers>
template<typename LayerType>
inline void AddVisitor<CustomLayers...>::operator()(LayerType* layer) const
{
  LayerAdd<LayerType>(layer);
}

template<typename... CustomLayers>
inline void AddVisitor<CustomLayers...>::operator()(MoreTypes layer) const
{
  layer.apply_visitor(*this);
}

template<typename... CustomLayers>
template<typename T>
inline typename std::enable_if<
    HasAddCheck<T, void(T::*)(LayerTypes<CustomLayers...>)>::value, void>::type
AddVisitor<CustomLayers...>::LayerAdd(T* layer) const
{
  layer->Add(newLayer);
}

template<typename... CustomLayers>
template<typename T>
inline typename std::enable_if<
    !HasAddCheck<T, void(T::*)(LayerTypes<CustomLayers...>)>::value, void>::type
AddVisitor<CustomLayers...>::LayerAdd(T* /* layer */) const
{
  /* Nothing to do here. */
}

} // namespace ann
} // namespace mlpack

#endif
