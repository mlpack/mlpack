/**
 * @file methods/decision_tree/select_functions/random_dimension_select.hpp
 * @author Ryan Curtin
 *
 * Selects one single random dimension to split on.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_RANDOM_DIMENSION_SELECT_HPP
#define MLPACK_METHODS_DECISION_TREE_RANDOM_DIMENSION_SELECT_HPP

namespace mlpack {

/**
 * This dimension selection policy only selects one single random dimension.
 */
class RandomDimensionSelect
{
 public:
  /**
   * Construct the RandomDimensionSelect object with the given number of
   * dimensions.
   */
  RandomDimensionSelect() : dimensions(0) { }

  /**
   * Get the first dimension to select from.
   */
  size_t Begin() const { return RandInt(dimensions); }

  /**
   * Get the last dimension to select from.
   */
  size_t End() const { return dimensions; }

  /**
   * Get the next (last) dimensions.  We only allow one dimension, so any 'next'
   * dimension is past our bounds.
   */
  size_t Next() const { return dimensions; }

  //! Get the number of dimensions.
  size_t Dimensions() const { return dimensions; }
  //! Set the number of dimensions.
  size_t& Dimensions() { return dimensions; }

 private:
  //! The number of dimensions to select from.
  size_t dimensions;
};

} // namespace mlpack

#endif
