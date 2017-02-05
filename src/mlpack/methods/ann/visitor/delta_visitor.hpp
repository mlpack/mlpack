/**
 * @file delta_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the Delta() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_DELTA_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_DELTA_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * DeltaVisitor exposes the delta parameter of the given module.
 */
class DeltaVisitor : public boost::static_visitor<arma::mat&>
{
 public:
  //! Return the delta parameter.
  template<typename LayerType>
  arma::mat& operator()(LayerType* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "delta_visitor_impl.hpp"

#endif
