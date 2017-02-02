/**
 * @file backward_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the Backward() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_BACKWARD_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_BACKWARD_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * BackwardVisitor executes the Backward() function given the input, error and
 * delta parameter.
 */
class BackwardVisitor : public boost::static_visitor<void>
{
 public:
  //! Execute the Backward() function given the input, error and delta
  //! parameter.
  BackwardVisitor(arma::mat&& input, arma::mat&& error, arma::mat&& delta);

  //! Execute the Backward() function.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The input parameter set.
  arma::mat&& input;

  //! The error parameter.
  arma::mat&& error;

  //! The delta parameter.
  arma::mat&& delta;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "backward_visitor_impl.hpp"

#endif
