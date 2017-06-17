/**
 * @file add_visitor_impl.hpp
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
template<typename T>
inline AddVisitor::AddVisitor(T newLayer) :
    newLayer(std::move(newLayer))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void AddVisitor::operator()(LayerType* layer) const
{
  LayerAdd<LayerType>(layer);
}

template<typename T>
inline typename std::enable_if<
    HasAddCheck<T, void(T::*)(LayerTypes)>::value, void>::type
AddVisitor::LayerAdd(T* layer) const
{
  layer->Add(newLayer);
}

template<typename T>
inline typename std::enable_if<
    !HasAddCheck<T, void(T::*)(LayerTypes)>::value, void>::type
AddVisitor::LayerAdd(T* /* layer */) const
{
  /* Nothing to do here. */
}

} // namespace ann
} // namespace mlpack

#endif
