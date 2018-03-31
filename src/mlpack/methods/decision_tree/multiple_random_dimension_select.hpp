/**
 * @file multiple_random_dimension_select.hpp
 * @author Ryan Curtin
 *
 * Select a number of random dimensions to pick from.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_MULTIPLE_RANDOM_DIMENSION_SPLIT_HPP
#define MLPACK_METHODS_DECISION_TREE_MULTIPLE_RANDOM_DIMENSION_SPLIT_HPP

namespace mlpack {
namespace tree {

/**
 * This dimension selection policy allows the selection from a few random
 * dimensions.  The number of random dimensions to use is specified by the
 * NumDimensions template parameter.
 *
 * @tparam NumDimensions Number of random dimensions to select.
 */
template<size_t NumDimensions = 3>
class MultipleRandomDimensionSelect
{
 public:
  /**
   * Instantiate the MultipleRandomDimensionSelect object.
   */
  MultipleRandomDimensionSelect(const size_t dimensions)
  {
    for (size_t i = 0; i < NumDimensions; ++i)
    {
      // Generate random different numbers.
      bool unique = false;
      size_t value;
      while (!unique)
      {
        value = math::RandInt(dimensions);

        // Check if we already have the value.
        unique = true;
        for (size_t j = 0; j < i; ++j)
        {
          if (values[j] == value)
          {
            unique = false;
            break;
          }
        }
      }

      values[i] = value;
    }

    values[NumDimensions] = std::numeric_limits<size_t>::max();
  }

  /**
   * Get the first random value.
   */
  size_t Begin()
  {
    i = 0;
    return values[0];
  }

  /**
   * Get the last random value.
   */
  size_t End() const { return size_t(-1); }

  /**
   * Get the next index.
   */
  size_t Next()
  {
    return values[++i];
  }

 private:
  //! The values we select from.
  size_t values[NumDimensions + 1];
  //! The current value we are looking at.
  size_t i;
};

} // namespace tree
} // namespace mlpack

#endif
