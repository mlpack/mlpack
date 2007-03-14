// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file la.h
 *
 * Core routines for la algebra.
 *
 * TODO: Currently lacking much, and eventually we want this to simply
 * use BLAS and LAPACK.
 */

#ifndef LINEAR_H
#define LINEAR_H

#include "matrix.h"

#include "base/scale.h"

#include <cmath>

/**
 * Namespace for linear-algebra helper routines.
 *
 * Will eventually include BLAS and LAPACk support.
 */
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
    DEBUG_ASSERT(x.length() == y.length());
    return DistanceSqEuclidean(x.length(), x.ptr(), y.ptr());
  }
  
  
  /* these will be replaced with lapack eventually */
#ifndef LAPACK
  /**
   * Finds the dot product of two vectors.
   */
  inline double VectorDot(index_t length, const double *x, const double *y) {
    double sum = 0;
    for (index_t i = 0; i < length; i++) {
      sum += x[i] * y[i];
    }
    return sum;
  }
  /**
   * Finds the dot product of two vectors.
   */
  inline double VectorDot(const Vector& x, const Vector& y) {
    DEBUG_ASSERT(x.length() == y.length());
    return VectorDot(x.length(), x.ptr(), y.ptr());
  }
  
  //---

  /**
   * Adds one vector to another (x <- x + y);
   */
  inline void VectorAddTo(index_t length, const double *x, double *y) {
    do {
      *y += *x;
      x++;
      y++;
    } while (--length);
  }
  
  /**
   * Adds one vector to another (x <- x + y);
   */
  inline void VectorAddTo(const Vector&x, Vector *y) {
    DEBUG_ASSERT(x.length() == y->length());
    VectorAddTo(x.length(), x.ptr(), y->ptr());
  }

  /**
   * Adds one matrix to another (X <- X + Y);
   */
  inline void MatrixAddTo(const Matrix& x, Matrix *y) {
    DEBUG_ASSERT(x.n_rows() == y->n_rows());
    DEBUG_ASSERT(x.n_cols() == y->n_cols());
    VectorAddTo(x.n_elements(), x.ptr(), y->ptr());
  }

  //---

  /**
   * Adds a vector times a factor to an existing vector (y <- y + xscale * x).
   */
  inline void VectorAddTo(index_t length,
      double xscale, const double *x, double *y) {
    do {
      *y += xscale * (*x);
      x++;
      y++;
    } while (--length);
  }

  /**
   * Adds a vector times a factor to an existing vector (y <- y + xscale * x).
   */
  inline void VectorAddTo(double xscale, const Vector&x, Vector *y) {
    DEBUG_ASSERT(x.length() == y->length());
    VectorAddTo(x.length(), xscale, x.ptr(), y->ptr());
  }

  /**
   * Adds a matrix times a factor to an existing matrix (Y <- Y + xscale * X).
   */
  inline void MatrixAddTo(double xscale, const Matrix& x, Matrix *y) {
    DEBUG_ASSERT(x.n_rows() == y->n_rows());
    DEBUG_ASSERT(x.n_cols() == y->n_cols());
    VectorAddTo(x.n_elements(), xscale, x.ptr(), y->ptr());
  }
  
  //---

  /**
   * Adds two vectors to a new vector (z <- x + y).
   */
  inline void VectorAddOverwrite(index_t length,
      const double *x, const double *y, double *c) {
    for (index_t i = 0; i < length; i++) {
      c[i] = x[i] + y[i];
    }
  }
  
  /**
   * Adds two vectors to a new vector (z <- x + y).
   * @param z will be Initialized to the sum of x and y
   */
  inline void VectorAddInit(
      const Vector& x, const Vector& y, Vector *z) {
    DEBUG_ASSERT(x.length() == y.length());
    z->Init(x.length());
    VectorAddOverwrite(x.length(), x.ptr(), y.ptr(), z->ptr());
  }
  
  /**
   * Adds two matrices to a new matrix (Z <- X + Y).
   * @param z will be Initialized to the sum of x and y
   */
  inline void MatrixAddInit(
      const Matrix& x, const Matrix& y, Matrix *z) {
    DEBUG_ASSERT(x.n_rows() == y.n_rows());
    DEBUG_ASSERT(x.n_cols() == y.n_cols());
    z->Init(x.n_rows(), x.n_cols());
    VectorAddOverwrite(x.n_elements(), x.ptr(), y.ptr(), z->ptr());
  }

  //---
  
  /**
   * Subtracts two vectors into a third (z <- x - y).
   */
  inline void VectorSubOverwrite(index_t length,
      const double *x, const double *y, double *z) {
    for (index_t i = 0; i < length; i++) {
      z[i] = x[i] - y[i];
    }
  }
  
  /**
   * Subtracts two vectors into a third (z <- x - y).
   * @param z will be Initialized to the sum of x and y
   */
  inline void VectorSubInit(
      const Vector& x, const Vector& y, Vector *z) {
    DEBUG_ASSERT(x.length() == y.length());
    z->Init(x.length());
    VectorSubOverwrite(x.length(), x.ptr(), y.ptr(), z->ptr());
  }

  //---

  /**
   * Subtracts a vector from an existing vector (y <- y - x).
   */
  inline void VectorSubFrom(index_t length, const double *x,
      double *y) {
    for (index_t i = 0; i < length; i++) {
      y[i] -= x[i];
    }
  }
    
  /**
   * Subtracts a vector from an existing vector (y <- y - x).
   */
  inline void VectorSubFrom(
      const Vector& x, Vector *y) {
    DEBUG_ASSERT(x.length() == y->length());
    VectorSubFrom(x.length(), x.ptr(), y->ptr());
  }

  //---
  
  /**
   * Multiplies a vector in-place by a scale (x <- alpha * x).
   */
  inline void VectorScale(index_t length, double alpha, double *x) {
    do {
      *x++ *= alpha;
    } while (--length);
  }

  /**
   * Multiplies a vector in-place by a scale (x <- alpha * x).
   */
  inline void VectorScale(double alpha, Vector *x) {
    VectorScale(x->length(), alpha, x->ptr());
  }

  /**
   * Multiplies a matrix in-place by a scale (X <- alpha * X).
   */
  inline void MatrixScale(double alpha, Matrix *x) {
    VectorScale(x->n_elements(), alpha, x->ptr());
  }
#endif
};

#endif
