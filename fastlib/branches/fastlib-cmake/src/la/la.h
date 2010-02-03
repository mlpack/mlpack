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
//#include "matrix.h"
//#include "uselapack.h"

#include "math/math_lib.h"

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
    DEBUG_SAME_SIZE(x.length(), y.length());
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
  /** Finds the trace of the matrix.
   *  Trace(A) is the sum of the diagonal elements
   */
  inline double Trace(Matrix &a) {
    // trace has meaning only for square matrices
    DEBUG_SAME_SIZE(a.n_cols(), a.n_rows());
    double trace=0;
    for(index_t i=0; i<a.n_cols(); i++) {
      trace+=a.get(i, i);
    }
    return trace;
  }
  
  /** Solves the classic least square problem y=x*a
   *  where y  is N x 1
   *        x  is N x m
   *        a  is m x 1
   *  We require that N >= m
   *  a should not be initialized
   */
  inline success_t LeastSquareFit(Vector &y, Matrix &x, Vector *a) {
    DEBUG_SAME_SIZE(y.length(), x.n_rows());
    DEBUG_ASSERT(x.n_rows() >= x.n_cols());
    Vector r_xy_vec;
    Matrix r_xx_mat;
    la::MulTransAInit(x, x, &r_xx_mat);
    la::MulInit(y, x, &r_xy_vec);
    success_t status = la::SolveInit(r_xx_mat, r_xy_vec, a);
    if unlikely(status != SUCCESS_PASS) {
      if (status==SUCCESS_FAIL) {
        FATAL("Least square fit failed \n");
      } else {
        NONFATAL("Least square fit returned a warning \n");
      }
    }
    return status;
  }

  /** Solves the classic least square problem y=x*a
   *  where y  is N x r
   *        x  is N x m
   *        a  is m x r
   *  We require that N >= m
   *  a should not be initialized
   */
  inline success_t LeastSquareFit(Matrix &y, Matrix &x, Matrix *a) {
    DEBUG_SAME_SIZE(y.n_rows(), x.n_rows());
    DEBUG_ASSERT(x.n_rows() >= x.n_cols());
    Matrix r_xy_mat;
    Matrix r_xx_mat;
    la::MulTransAInit(x, x, &r_xx_mat);
    la::MulTransAInit(x, y, &r_xy_mat);
    success_t status = la::SolveInit(r_xx_mat, r_xy_mat, a);
    if unlikely(status != SUCCESS_PASS) {
      if (status==SUCCESS_FAIL) {
        FATAL("Least square fit failed \n");
      } else {
        NONFATAL("Least square fit returned a warning \n");
      }
    }
    return status;
  }
  
  /** Solves the classic least square problem y=x'*a
   *  where y  is N x r
   *        x  is m x N
   *        a  is m x r
   *  We require that N >= m
   *  a should not be initialized
   */
  inline success_t LeastSquareFitTrans(Matrix &y, Matrix &x, Matrix *a) {
    DEBUG_SAME_SIZE(y.n_rows(), x.n_cols());
    DEBUG_ASSERT(x.n_cols() >= x.n_rows());
    Matrix r_xy_mat;
    Matrix r_xx_mat;
    la::MulTransBInit(x, x, &r_xx_mat);
    la::MulInit(x, y, &r_xy_mat);
    success_t status = la::SolveInit(r_xx_mat, r_xy_mat, a);
    if unlikely(status != SUCCESS_PASS) {
      if (status==SUCCESS_FAIL) {
        FATAL("Least square fit failed \n");
      } else {
        NONFATAL("Least square fit returned a warning \n");
      }
    }
    return status;
  }
};

#endif
