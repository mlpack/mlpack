/**
 * @file methods/sparse_coding/nothing_initializer.hpp
 * @author Ryan Curtin
 *
 * An initializer for SparseCoding which does precisely nothing.  It is useful
 * for when you have an already defined dictionary and you plan on setting it
 * with SparseCoding::Dictionary().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_CODING_NOTHING_INITIALIZER_HPP
#define MLPACK_METHODS_SPARSE_CODING_NOTHING_INITIALIZER_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * A DictionaryInitializer for SparseCoding which does not initialize anything;
 * it is useful for when the dictionary is already known and will be set with
 * SparseCoding::Dictionary().
 */
class NothingInitializer
{
 public:
  /**
   * This function does not initialize the dictionary.  This will cause problems
   * for SparseCoding if the dictionary is not set manually before running the
   * method.
   */
  template<typename MatType>
  static void Initialize(const MatType& /* data */,
                         const size_t /* atoms */,
                         MatType& /* dictionary */)
  {
    // Do nothing!
  }
};

} // namespace mlpack

#endif
