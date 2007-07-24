// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file la.h
 *
 * Core routines for linear algebra.
 *
 * See uselapack.h for more linear algebra routines.
 */

#ifndef LINEAR_H
#define LINEAR_H

#include "matrix.h"
#include "uselapack.h"

#include "base/scale.h"

#include <math.h>

namespace la {
  /**
   * Finds the Euclidean distance squared between two vectors.
   */
  inline double DistanceSqEuclidean(
      index_t length, const double *va, const double *vb) {
    // TODO: Use blas to do this
    double s = 0;
    
    do {
      --length;
      double d = *va++ - *vb++;
      s += d * d;
    } while (length);
    
    return s;
  }
  /**
   * Finds the Euclidean distance squared between two vectors.
   */
  inline double DistanceSqEuclidean(const Vector& x, const Vector& y) {
    DEBUG_SAME_INT(x.length(), y.length());
    return DistanceSqEuclidean(x.length(), x.ptr(), y.ptr());
  }
};

#endif
