/**
 * @file deterministic_set_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Deterministic() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_DETERMINISTIC_SET_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_DETERMINISTIC_SET_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "deterministic_set_visitor.hpp"

namespace mlpack {
namespace ann {

//! DeterministicSetVisitor visitor class.
inline DeterministicSetVisitor::DeterministicSetVisitor(
    const bool deterministic) : deterministic(deterministic)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void DeterministicSetVisitor::operator()(LayerType* layer) const
{
  LayerDeterministic(layer);
}

template<typename T>
inline typename std::enable_if<
    HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
DeterministicSetVisitor::LayerDeterministic(T* layer) const
{
  layer->Deterministic() = deterministic;

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(DeterministicSetVisitor(deterministic),
        layer->Model()[i]);
  }
}

template<typename T>
inline typename std::enable_if<
    !HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
DeterministicSetVisitor::LayerDeterministic(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(DeterministicSetVisitor(deterministic),
        layer->Model()[i]);
  }
}

template<typename T>
inline typename std::enable_if<
    HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
DeterministicSetVisitor::LayerDeterministic(T* layer) const
{
  layer->Deterministic() = deterministic;
}

template<typename T>
inline typename std::enable_if<
    !HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
DeterministicSetVisitor::LayerDeterministic(T* /* input */) const
{
  /* Nothing to do here. */
}

} // namespace ann
} // namespace mlpack

#endif
