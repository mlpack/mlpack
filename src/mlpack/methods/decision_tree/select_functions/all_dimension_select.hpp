/**
 * @file methods/decision_tree/select_functions/all_dimension_select.hpp
 * @author Ryan Curtin
 *
 * Selects all dimensions for a split.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_ALL_DIMENSION_SELECT_HPP
#define MLPACK_METHODS_DECISION_TREE_ALL_DIMENSION_SELECT_HPP

namespace mlpack {

/**
 * This dimension selection policy allows any dimension to be selected for
 * splitting.
 */
class AllDimensionSelect
{
 public:
  /**
   * Construct the AllDimensionSelect object for the given number of dimensions.
   */
  AllDimensionSelect() : i(0), dimensions(0) { }

  /**
   * Get the first dimension to select from.
   */
  size_t Begin()
  {
    i = 0;
    return 0;
  }

  /**
   * Get the last dimension to select from.
   */
  size_t End() const { return dimensions; }

  /**
   * Get the next dimension.
   */
  size_t Next() { return ++i; }

  //! Get the number of dimensions.
  size_t Dimensions() const { return dimensions; }
  //! Modify the number of dimensions.
  size_t& Dimensions() { return dimensions; }

 private:
  //! The current dimension we are looking at.
  size_t i;
  //! The number of dimensions to select from.
  size_t dimensions;
};

} // namespace mlpack

#endif
