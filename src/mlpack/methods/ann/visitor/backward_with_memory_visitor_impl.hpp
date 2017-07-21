/**
 * @file backward_with_memory_visitor_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the BackwardWithMemory() function(which accepts memory)
 * layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_BACKWARD_WITH_MEMORY_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_BACKWARD_WITH_MEMORY_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "Backward_with_memory_visitor.hpp"

namespace mlpack {
namespace ann {

//! BackwardWithMemoryVisitor visitor class.
inline BackwardWithMemoryVisitor::BackwardWithMemoryVisitor(arma::mat&& input,
                                                            arma::mat&& memory,
                                                            arma::mat&& error,
                                                            arma::mat&& delta) :
    input(std::move(input)),
    memory(std::move(memory)),
    output(std::move(error)),
    delta(std::move(delta))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void BackwardWithMemoryVisitor::operator()(LayerType* layer) const
{
  BackwardWithMemory(layer);
}

template<typename T>
inline typename std::enable_if<
    HasBackwardWithMemoryCheck<T, void(T::*)()>::value, void>::type
BackwardWithMemoryVisitor::BackwardWithMemory(T* layer) const
{
  layer->BackwardWithMemory(std::move(input),
                           std::move(memory),
                           std::move(output));
}

template<typename T>
inline typename std::enable_if<
    !HasBackwardWithMemoryCheck<T, void(T::*)()>::value, void>::type
BackwardWithMemoryVisitor::BackwardWithMemory(T* /* layer */) const
{
  /* Nothing to do here. */
}

} // namespace ann
} // namespace mlpack

#endif
