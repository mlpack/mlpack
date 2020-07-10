/**
 * @file methods/ann/visitor/reset_cell_visitor_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the ResetCell() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_RESET_CELL_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_RESET_CELL_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "reset_cell_visitor.hpp"

namespace mlpack {
namespace ann {

//! ResetVisitor visitor class.
inline ResetCellVisitor::ResetCellVisitor(const size_t size) : size(size)
{
  /* Nothing to do here. */
}

//! ResetVisitor visitor class.
template<typename LayerType>
inline void ResetCellVisitor::operator()(LayerType* layer) const
{
  ResetCell(layer);
}

inline void ResetCellVisitor::operator()(MoreTypes layer) const
{
  layer.apply_visitor(*this);
}

template<typename T>
inline typename std::enable_if<
    HasResetCellCheck<T, void(T::*)(const size_t)>::value, void>::type
ResetCellVisitor::ResetCell(T* layer) const
{
  layer->ResetCell(size);
}

template<typename T>
inline typename std::enable_if<
    !HasResetCellCheck<T, void(T::*)(const size_t)>::value, void>::type
ResetCellVisitor::ResetCell(T* /* layer */) const
{
  /* Nothing to do here. */
}

} // namespace ann
} // namespace mlpack

#endif
