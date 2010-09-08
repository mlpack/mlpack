/**
 * @file discrete.cc
 *
 * Helpers for discrete math (implementation).
 */

#include "discrete.h"
#include <stdlib.h>

double core::math::BinomialCoefficient(int n, int k) {
  int n_k = n - k;
  double nchsk = 1;
  int i;

  if (k > n || k < 0) {
    return 0;
  }

  if (k < n_k) {
    k = n_k;
    n_k = n - k;
  }

  for (i = 1; i <= n_k; i++) {
    nchsk *= (++k);
    nchsk /= i;
  }
  return nchsk;
}

double core::math::Factorial(int d) {
  double v = 1;

  for (int i = 2; i <= d; i++) {
    v *= i;
  }

  return v;
}
