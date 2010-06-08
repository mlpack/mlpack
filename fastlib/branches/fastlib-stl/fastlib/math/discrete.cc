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
 * @file discrete.cc
 *
 * Helpers for discrete math (implementation).
 */

#include "discrete.h"
//#include "discrete.h"

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
  
  DEBUG_ASSERT(d >= 0);
  
  for (int i = 2; i <= d; i++) {
    v *= i;
  }
  
  return v;
}

void math::MakeIdentityPermutation(index_t size, index_t *array) {
  for (index_t i = 0; i < size; i++) {
    array[i] = i;
  }
}

void math::MakeIdentityPermutation(index_t size, std::vector<index_t>& result) {
  result.reserve(size);

  for (index_t i = 0; i < size; i++) 
    result.push_back(i);
}
void math::MakeRandomPermutation(index_t size, index_t *array) {
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

void math::MakeInversePermutation(index_t size,
    const index_t *original, index_t *reverse) {
  for (index_t i = 0; i < size; i++) {
    reverse[original[i]] = i;
  }
}
