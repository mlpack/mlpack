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

#include "fastlib/math/math_lib.h"

#include <math.h>

namespace la {
  /**
   * Finds the Euclidean distance squared between two vectors.
   */
  template<typename Precision>
  inline Precision DistanceSqEuclidean(
      index_t length, const Precision *va, const Precision *vb) {
    Precision s = 0;
    do {
      Precision d = *va++ - *vb++;
      s += d * d;
    } while (--length);
    return s;
  }
  /**
   * Finds the Euclidean distance squared between two vectors.
   */
  template<typename Precision>
  inline Precision DistanceSqEuclidean(const GenVector<Precision>& x, 
                                       const GenVector<Precision>& y) {
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
  template<typename Precision, int t_pow>
  inline Precision RawLMetric(
      index_t length, const Precision *va, const Precision *vb) {
    Precision s = 0;
    do {
      Precision d = *va++ - *vb++;
      s += math::PowAbs<Precision, t_pow, 1>(d);
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
  template<typename Precision, int t_pow>
  inline Precision LMetric(
      index_t length, const Precision *va, const Precision *vb) {
    return math::Pow<Precision, 1, t_pow>(RawLMetric<Precision, t_pow>(length, va, vb));
  }
  /** Finds the trace of the matrix.
   *  Trace(A) is the sum of the diagonal elements
   */
  template<typename Precision>
  inline Precision Trace(GenMatrix<Precision> &a) {
    // trace has meaning only for square matrices
    DEBUG_SAME_SIZE(a.n_cols(), a.n_rows());
    Precision trace=0;
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
  template<typename Precision>
  inline success_t LeastSquareFit(GenVector<Precision> &y, 
      GenMatrix<Precision> &x, GenVector<Precision> *a) {
    DEBUG_SAME_SIZE(y.length(), x.n_rows());
    DEBUG_ASSERT(x.n_rows() >= x.n_cols());
    GenVector<Precision> r_xy_vec;
    GenMatrix<Precision> r_xx_mat;
    la::MulTransAInit<Precision>(x, x, &r_xx_mat);
    la::MulInit<Precision>(y, x, &r_xy_vec);
    success_t status = la::SolveInit<Precision>(r_xx_mat, r_xy_vec, a);
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
  template<typename Precision>
  inline success_t LeastSquareFit(GenMatrix<Precision> &y, 
      GenMatrix<Precision> &x, GenMatrix<Precision> *a) {
    DEBUG_SAME_SIZE(y.n_rows(), x.n_rows());
    DEBUG_ASSERT(x.n_rows() >= x.n_cols());
    GenMatrix<Precision> r_xy_mat;
    GenMatrix<Precision> r_xx_mat;
    la::MulTransAInit<Precision>(x, x, &r_xx_mat);
    la::MulTransAInit<Precision>(x, y, &r_xy_mat);
    success_t status = la::SolveInit<Precision>(r_xx_mat, r_xy_mat, a);
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
  template<typename Precision>
  inline success_t LeastSquareFitTrans(GenMatrix<Precision> &y, 
      GenMatrix<Precision> &x, GenMatrix<Precision> *a) {
    DEBUG_SAME_SIZE(y.n_rows(), x.n_cols());
    DEBUG_ASSERT(x.n_cols() >= x.n_rows());
    GenMatrix<Precision> r_xy_mat;
    GenMatrix<Precision> r_xx_mat;
    la::MulTransBInit<Precision>(x, x, &r_xx_mat);
    la::MulInit<Precision>(x, y, &r_xy_mat);
    success_t status = la::SolveInit<Precision>(r_xx_mat, r_xy_mat, a);
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
