/**
 * @file reset_cell_visitor.hpp
 * @author Sumedh Ghaisas
 *
 * Boost static visitor abstraction for calling ResetCell function on RNN cells. 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_RESET_CELL_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_RESET_CELL_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * ResetCellVisitor executes the ResetCell() function.
 */
class ResetCellVisitor : public boost::static_visitor<void>
{
 public:
  //! Reset the cell using the given size.
  ResetCellVisitor(const size_t size);

  //! Execute the ResetCell() function.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

  void operator()(MoreTypes layer) const;

 private:
  size_t size;

  //! Execute the ResetCell() function for a module which implements
  //! the ResetCell() function.
  template<typename T>
  typename std::enable_if<
      HasResetCellCheck<T, void(T::*)(const size_t)>::value, void>::type
  ResetCell(T* layer) const;

  //! Do not execute the Reset() function for a module which doesn't implement
  // the Reset() or Model() function.
  template<typename T>
  typename std::enable_if<
      !HasResetCellCheck<T, void(T::*)(const size_t)>::value, void>::type
  ResetCell(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "reset_cell_visitor_impl.hpp"

#endif
