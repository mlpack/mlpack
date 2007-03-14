// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file discrete.cc
 *
 * Helpers for discrete math (implementation).
 */

#include "discrete.h"

#include <stdlib.h>

namespace math {

double Factorial(int d) {
  double v = 1;
  
  DEBUG_ASSERT(d >= 0);
  
  for (int i = 2; i <= d; i++) {
    v *= i;
  }
  
  return v;
}

void MakeIdentityPermutation(index_t size, index_t *array) {
  for (index_t i = 0; i < size; i++) {
    array[i] = i;
  }
}

void MakeRandomPermutation(index_t size, index_t *array) {
  // Regular permutation algorithm.
  // This is cache inefficient for large sizes; large caches might
  // warrant a more sophisticated blocked algorithm.
  
  if (unlikely(size == 0)) {
    return;
  }
  
  array[0] = 0;
  
  for (index_t i = 1; i < size; i++) {
    index_t victim = rand() % i;
    array[i] = array[victim];
    array[victim] = i;
  }
}

};
