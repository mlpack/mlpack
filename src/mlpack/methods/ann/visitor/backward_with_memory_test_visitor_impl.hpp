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
#ifndef MLPACK_METHODS_ANN_VISITOR_BACKWARD_WITH_MEMORY_TEST_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_BACKWARD_WITH_MEMORY_TEST_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "backward_with_memory_test_visitor.hpp"

namespace mlpack {
namespace ann {

inline BackwardWithMemoryTestVisitor::BackwardWithMemoryTestVisitor(arma::mat&& output,
                                                            arma::mat&& memory,
                                                            arma::mat&& error,
                                                            arma::mat&& delta,
                                                            arma::mat&& dM) :
    output(std::move(output)),
    memory(std::move(memory)),
    error(std::move(error)),
    delta(std::move(delta)),
    dM(dM)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void BackwardWithMemoryTestVisitor::operator()(LayerType* layer) const
{
  BackwardWithMemoryTest(layer);
}

template<typename T>
inline typename std::enable_if<
    HasBackwardWithMemoryTestCheck<T, void(T::*)(const arma::mat&&,
    const arma::mat&&, arma::mat&&, arma::mat&&,
    arma::mat&&)>::value, void>::type
BackwardWithMemoryTestVisitor::BackwardWithMemoryTest(T* layer) const
{
  layer->BackwardWithMemoryTest(std::move(output),
                                std::move(memory),
                                std::move(error),
                                std::move(delta),
                                std::move(dM));
}

template<typename T>
inline typename std::enable_if<
    !HasBackwardWithMemoryTestCheck<T, void(T::*)(const arma::mat&&,
    const arma::mat&&, arma::mat&&, arma::mat&&,
    arma::mat&&)>::value, void>::type
BackwardWithMemoryTestVisitor::BackwardWithMemoryTest(T* /* layer */) const
{
  /* Nothing to do here. */
}

} // namespace ann
} // namespace mlpack

#endif
