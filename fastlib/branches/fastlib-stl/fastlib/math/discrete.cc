/**
 * @file discrete.cc
 *
 * Helpers for discrete math (implementation).
 */

#include "discrete.h"
#include "fastlib/fx/io.h"

#include <stdlib.h>

double math::BinomialCoefficient(int n, int k) {
  int n_k = n - k;
  double nchsk = 1;
  int i;
  
  if(k > n || k < 0) {
    return 0;
  }
  
  if(k < n_k) {
    k = n_k;
    n_k = n - k;
  }
  
  for(i = 1; i <= n_k; i++) {
    nchsk *= (++k);
    nchsk /= i;
  }
  return nchsk;
}

double math::Factorial(int d) {
  double v = 1;
  
  mlpack::IO::Assert(d >= 0);
  
  for (int i = 2; i <= d; i++) {
    v *= i;
  }
  
  return v;
}

void math::MakeIdentityPermutation(size_t size, size_t *array) {
  for (size_t i = 0; i < size; i++) {
    array[i] = i;
  }
}

void math::MakeIdentityPermutation(size_t size, std::vector<size_t>& result) {
  result.reserve(size);

  for (size_t i = 0; i < size; i++) 
    result.push_back(i);
}
void math::MakeRandomPermutation(size_t size, size_t *array) {
  // Regular permutation algorithm.
  // This is cache inefficient for large sizes; large caches might
  // warrant a more sophisticated blocked algorithm.
  
  if (size == 0) {
    return;
  }
  
  array[0] = 0;
  
  for (size_t i = 1; i < size; i++) {
    size_t victim = rand() % i;
    array[i] = array[victim];
    array[victim] = i;
  }
}

void math::MakeInversePermutation(size_t size,
    const size_t *original, size_t *reverse) {
  for (size_t i = 0; i < size; i++) {
    reverse[original[i]] = i;
  }
}
