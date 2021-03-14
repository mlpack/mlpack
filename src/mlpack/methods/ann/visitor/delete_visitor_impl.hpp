/**
 * @file methods/ann/visitor/delete_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Delete() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_DELETE_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_DELETE_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "delete_visitor.hpp"

namespace mlpack {
namespace ann {

//! DeleteVisitor visitor class.
template<typename LayerType>
inline typename std::enable_if<
    !HasModelCheck<LayerType>::value, void>::type
DeleteVisitor::operator()(LayerType* layer) const
{
  if (layer)
    delete layer;
}

template<typename LayerType>
inline typename std::enable_if<
    HasModelCheck<LayerType>::value, void>::type
DeleteVisitor::operator()(LayerType* layer) const
{
  if (layer)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
      boost::apply_visitor(DeleteVisitor(), layer->Model()[i]);

    delete layer;
  }
}

inline void DeleteVisitor::operator()(MoreTypes layer) const
{
  layer.apply_visitor(*this);
}

} // namespace ann
} // namespace mlpack

#endif
