// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file statistics.cc
 *
 * Implementation for statistics helpers.
 */

#include "statistics.h"
#include "math.h"

namespace math {

double Mean(Vector V) {
  double c = 0.0;
  index_t n = V.length();
  for (index_t i=0; i<n; i++)
    c = c + V[i];
  return c/n;
}

};
