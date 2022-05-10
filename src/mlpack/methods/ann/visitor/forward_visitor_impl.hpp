/**
 * @file methods/ann/visitor/forward_visitor_impl.hpp
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
inline ForwardVisitor::ForwardVisitor(const arma::mat& input, arma::mat& output) :
    input(input),
    output(output)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void ForwardVisitor::operator()(LayerType* layer) const
{
  layer->Forward(input, output);
}

inline void ForwardVisitor::operator()(MoreTypes layer) const
{
  layer.apply_visitor(*this);
}

} // namespace ann
} // namespace mlpack

#endif
