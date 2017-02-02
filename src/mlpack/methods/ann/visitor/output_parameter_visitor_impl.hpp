/**
 * @file output_parameter_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the OutputParameter() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_OUTPUT_PARAMETER_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_OUTPUT_PARAMETER_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "output_parameter_visitor.hpp"

namespace mlpack {
namespace ann {

//! OutputParameterVisitor visitor class.
template<typename LayerType>
inline arma::mat& OutputParameterVisitor::operator()(LayerType *layer) const
{
  return layer->OutputParameter();
}

} // namespace ann
} // namespace mlpack

#endif
