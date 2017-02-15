/**
 * @file forward_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Forward() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_FORWARD_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_FORWARD_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "forward_visitor.hpp"

namespace mlpack {
namespace ann {

//! ForwardVisitor visitor class.
inline ForwardVisitor::ForwardVisitor(arma::mat&& input, arma::mat&& output) :
  input(std::move(input)),
  output(std::move(output))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void ForwardVisitor::operator()(LayerType* layer) const
{
  layer->Forward(std::move(input), std::move(output));
}

} // namespace ann
} // namespace mlpack

#endif
