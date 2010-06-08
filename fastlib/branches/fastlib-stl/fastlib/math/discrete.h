/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file discrete.h
 *
 * Discrete math helpers.
 */

#ifndef MATH_DISCRETE_H
#define MATH_DISCRETE_H

#include "../base/base.h"

#include <vector>

#include <math.h>

namespace math {
  /**
   * Computes the factorial of an integer.
   */
  __attribute__((const)) double Factorial(int d);
  
  /**
   * Computes the binomial coefficient, n choose k for nonnegative integers
   * n and k
   *
   * @param n the first nonnegative integer argument
   * @param k the second nonnegative integer argument
   * @return the binomial coefficient n choose k
   */
  __attribute__((const)) double BinomialCoefficient(int n, int k);

  /**
   * Creates an identity permutation where the element i equals i.
   *
   * Low-level pointer version -- preferably use the @c ArrayList
   * version instead.
   *
   * For instance, result[0] == 0, result[1] == 1, result[2] == 2, etc.
   *
   * @param size the number of elements in the permutation
   * @param array a place to store the permutation
   */
  void MakeIdentityPermutation(index_t size, std::vector<index_t>::iterator array);
  
  /**
   * Creates an identity permutation where the element i equals i.
   *
   * For instance, result[0] == 0, result[1] == 1, result[2] == 2, etc.
   *
   * @param size the size to initialize the result to
   * @param result will be initialized to the identity permutation
   */
  inline void MakeIdentityPermutation(
      index_t size, std::vector<index_t>& result) {
    result.reserve(size);
    MakeIdentityPermutation(size, result.begin());
  }
  
  /**
   * Creates a random permutation and stores it in an existing C array
   * (power user version).
   *
   * The random permutation is over the integers 0 through size - 1.
   *
   * @param size the number of elements
   * @param array the array to store a permutation in
   */
  void MakeRandomPermutation(index_t size, std::vector<index_t>& array);
  
  /**
   * Creates a random permutation over integers 0 throush size - 1.
   *
   * @param size the number of elements
   * @param result will be initialized to a permutation array
   */
  inline void MakeRandomPermutation(
      index_t size, std::vector<index_t>& result) {
    result.reserve(size);
    MakeRandomPermutation(size, result);
  }

  /**
   * Inverts or transposes an existing permutation.
   */
  void MakeInversePermutation(index_t size,
      const std::vector<index_t>& original, std::vector<index_t>& reverse);

  /**
   * Inverts or transposes an existing permutation.
   */
  inline void MakeInversePermutation(
      const std::vector<index_t>& original, std::vector<index_t>& reverse) {
    reverse.reserve(original.size());
    MakeInversePermutation(original.size(), original, reverse);
  }
  
  template<typename TAnyIntegerType>
  inline bool IsPowerTwo(TAnyIntegerType i) {
    return (i & (i - 1)) == 0;
  }
  
  /**
   * Finds the log base 2 of an integer.
   *
   * This integer must absolutely be a power of 2.
   */
  inline unsigned IntLog2(unsigned i) {
    unsigned l;
    for (l = 0; (unsigned(1) << l) != i; l++) {
      DEBUG_ASSERT_MSG(l < 1024, "Taking IntLog2 of a non-power-of-2: %u.", i);
    }
    return l;
  }
};

#endif
