// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file discrete.h
 *
 * Discrete math helpers.
 */

#ifndef MATH_DISCRETE_H
#define MATH_DISCRETE_H

#include "base/base.h"
#include "col/arraylist.h"

#include <math.h>

namespace math {
  /**
   * Computes the factorial of an integer.
   */
  COMPILER_FUNCTIONAL
  double Factorial(int d);
  
  /**
   * Computes the binomial coefficient, n choose k for nonnegative integers
   * n and k
   *
   * @param n the first nonnegative integer argument
   * @param k the second nonnegative integer argument
   * @return the binomial coefficient n choose k
   */
  COMPILER_FUNCTIONAL
  double BinomialCoefficient(int n, int k);

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
  void MakeIdentityPermutation(index_t size, index_t *array);
  
  /**
   * Creates an identity permutation where the element i equals i.
   *
   * For instance, result[0] == 0, result[1] == 1, result[2] == 2, etc.
   *
   * @param size the size to initialize the result to
   * @param result will be initialized to the identity permutation
   */
  inline void MakeIdentityPermutation(
      index_t size, ArrayList<index_t> *result) {
    result->Init(size);
    MakeIdentityPermutation(size, result->begin());
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
  void MakeRandomPermutation(index_t size, index_t *array);
  
  /**
   * Creates a random permutation over integers 0 throush size - 1.
   *
   * @param size the number of elements
   * @param result will be initialized to a permutation array
   */
  inline void MakeRandomPermutation(
      index_t size, ArrayList<index_t> *result) {
    result->Init(size);
    MakeRandomPermutation(size, result->begin());
  }

  /**
   * Inverts or transposes an existing permutation.
   */
  void MakeInversePermutation(index_t size,
      const index_t *original, index_t *reverse);

  /**
   * Inverts or transposes an existing permutation.
   */
  inline void MakeInversePermutation(
      const ArrayList<index_t>& original, ArrayList<index_t> *reverse) {
    reverse->Init(original.size());
    MakeInversePermutation(original.size(), original.begin(), reverse->begin());
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
