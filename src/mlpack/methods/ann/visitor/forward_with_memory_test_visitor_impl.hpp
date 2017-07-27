/**
 * @file forward_with_memory_visitor_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the Forward() function(which accepts memory)
 * layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_FORWARD_WITH_MEMORY_TEST_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_FORWARD_WITH_MEMORY_TEST_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "forward_with_memory_test_visitor.hpp"

namespace mlpack {
namespace ann {

//! ForwardWithMemoryVisitor visitor class.
inline ForwardWithMemoryTestVisitor::ForwardWithMemoryTestVisitor(
  arma::mat&& input,
  arma::mat&& memory,
  arma::mat&& output) :
    input(std::move(input)),
    memory(std::move(memory)),
    output(std::move(output))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void ForwardWithMemoryTestVisitor::operator()(LayerType* layer) const
{
  ForwardWithMemoryTest(layer);
}

template<typename T>
inline typename std::enable_if<
    HasForwardWithMemoryTestCheck<T, void(T::*)(arma::mat&&, const arma::mat&&,
    arma::mat&&)>::value, void>::type
ForwardWithMemoryTestVisitor::ForwardWithMemoryTest(T* layer) const
{
  layer->ForwardWithMemoryTest(std::move(input),
                               std::move(memory),
                               std::move(output));
}

template<typename T>
inline typename std::enable_if<
    !HasForwardWithMemoryTestCheck<T, void(T::*)(arma::mat&&, const arma::mat&&,
    arma::mat&&)>::value, void>::type
ForwardWithMemoryTestVisitor::ForwardWithMemoryTest(T* /* layer */) const
{
  /* Nothing to do here. */
}

} // namespace ann
} // namespace mlpack

#endif
