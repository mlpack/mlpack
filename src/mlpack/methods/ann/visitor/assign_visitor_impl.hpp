/**
 * @file assign_visitor_impl.hpp
 * @author Shangtong Zhang
 *
 * This file provides an implementation for assignment between layers
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_VISITOR_ASSIGN_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_ASSIGN_VISITOR_IMPL_HPP

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

template <typename LayerType>
inline void AssignVisitor::operator () (LayerType* layer1, LayerType* layer2) const
{
  *layer1 = *layer2;
}

template <typename LayerType1, typename LayerType2>
inline void AssignVisitor::operator () (LayerType1*, LayerType2*) const
{
  Log::Fatal << "Incompatible layer can't be assigned to each other." << std::endl;
};

} // namespace ann
} // namespace mlpack

#endif

