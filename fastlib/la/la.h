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

#include "math/math.h"

#include <math.h>

namespace la {
  /**
   * Finds the Euclidean distance squared between two vectors.
   */
  inline double DistanceSqEuclidean(
      index_t length, const double *va, const double *vb) {
    double s = 0;
    do {
      double d = *va++ - *vb++;
      s += d * d;
    } while (--length);
    return s;
  }
  /**
   * Finds the Euclidean distance squared between two vectors.
   */
  inline double DistanceSqEuclidean(const Vector& x, const Vector& y) {
    DEBUG_ASSERT_INDICES_EQUAL(x.length(), y.length());
    return DistanceSqEuclidean(x.length(), x.ptr(), y.ptr());
  }
  /**
   * Finds an L_p metric distance except doesn't perform the root
   * at the end.
   *
   * @param t_pow the power each distance calculatin is raised to
   * @param length the length of the vectors
   * @param va first vector
   * @param vb second vector
   */
  template<int t_pow>
  inline double RawLMetric(
      index_t length, const double *va, const double *vb) {
    double s = 0;
    do {
      double d = *va++ - *vb++;
      s += math::PowAbs<t_pow, 1>(d);
    } while (--length);
    return s;
  }
  /**
   * Finds an L_p metric distance AND performs the root
   * at the end.
   *
   * @param t_pow the power each distance calculatin is raised to
   * @param length the length of the vectors
   * @param va first vector
   * @param vb second vector
   */
  template<int t_pow>
  inline double LMetric(
      index_t length, const double *va, const double *vb) {
    return math::Pow<1, t_pow>(RawLMetric<t_pow>(length, va, vb));
  }
};

#endif
