/**
 * @file forward_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the Forward() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_FORWARD_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_FORWARD_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * ForwardVisitor executes the Forward() function given the input and output
 * parameter.
 */
class ForwardVisitor : public boost::static_visitor<void>
{
 public:
  //! Execute the Foward() function given the input and output parameter.
  ForwardVisitor(arma::mat&& input, arma::mat&& output);

  //! Execute the Foward() function.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The input parameter set.
  arma::mat&& input;

  //! The output parameter set.
  arma::mat&& output;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "forward_visitor_impl.hpp"

#endif
