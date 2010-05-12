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
 * @file kernel.h
 *
 * Common statistical kernels.
 */

#ifndef MATH_KERNEL_H
#define MATH_KERNEL_H

#include "../base/base.h"
#include "geometry.h"
#include "math_lib.h"

#include <math.h>

/* More to come soon - Gaussian, Epanechnakov, etc. */

/**
 * Standard multivariate Gaussian kernel.
 *
 */
class GaussianKernel {
 private:
  double neg_inv_bandwidth_2sq_;
  double bandwidth_sq_;

 /* OBJECT_TRAVERSAL(GaussianKernel) {
    OT_OBJ(neg_inv_bandwidth_2sq_);
    OT_OBJ(bandwidth_sq_);
  }
 */

 public:

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & neg_inv_bandwidth_2sq_;
    ar & bandwidth_sq_;
  }

  static const bool HAS_CUTOFF = false;

 public:
  double bandwidth_sq() const {
    return bandwidth_sq_;
  }

  void Init(double bandwidth_in, index_t dims) {
    Init(bandwidth_in);
  }

  /**
   * Initializes to a specific bandwidth.
   *
   * @param bandwidth_in the standard deviation sigma
   */
  void Init(double bandwidth_in) {
    bandwidth_sq_ = bandwidth_in * bandwidth_in;
    neg_inv_bandwidth_2sq_ = -1.0 / (2.0 * bandwidth_sq_);
  }

  /**
   * Evaluates an unnormalized density, given the distance between
   * the kernel's mean and a query point.
   */
  double EvalUnnorm(double dist) const {
    return EvalUnnormOnSq(dist * dist);
  }

  /**
   * Evaluates an unnormalized density, given the square of the
   * distance.
   */
  double EvalUnnormOnSq(double sqdist) const {
    double d = exp(sqdist * neg_inv_bandwidth_2sq_);
    return d;
  }

  /** Unnormalized range on a range of squared distances. */
  DRange RangeUnnormOnSq(const DRange& range) const {
    return DRange(EvalUnnormOnSq(range.hi), EvalUnnormOnSq(range.lo));  //!! TODO explain
  }

  /**
   * Gets the maximum unnormalized value.
   */
  double MaxUnnormValue() {
    return 1;
  }

  /**
   * Divide by this constant when you're done.
   */
  double CalcNormConstant(index_t dims) const {
    // Changed because * faster than / and 2 * math::PI opt out.  RR
    //return pow((-math::PI/neg_inv_bandwidth_2sq_), dims/2.0);
    return pow(2 * math::PI * bandwidth_sq_, dims / 2.0);
  }
};

/**
 * Standard multivariate Gaussian kernel.
 *
 */
class GaussianStarKernel {
 private:
  double neg_inv_bandwidth_2sq_;
  double factor_;
  double bandwidth_sq_;
  double critical_point_sq_;
  double critical_point_value_;
  
  OBJECT_TRAVERSAL(GaussianStarKernel) {
    OT_OBJ(neg_inv_bandwidth_2sq_);
    OT_OBJ(factor_);
    OT_OBJ(bandwidth_sq_);
    OT_OBJ(critical_point_sq_);
    OT_OBJ(critical_point_value_);
  }
  
 public:
  static const bool HAS_CUTOFF = false;
 
 public:
  double bandwidth_sq() const {
    return bandwidth_sq_;
  }

  /**
   * Initializes to a specific bandwidth.
   *
   * @param bandwidth_in the standard deviation sigma
   */
  void Init(double bandwidth_in, index_t dims) {
    bandwidth_sq_ = bandwidth_in * bandwidth_in;
    neg_inv_bandwidth_2sq_ = -1.0 / (2.0 * bandwidth_sq_);
    factor_ = pow(2.0, -dims / 2.0 - 1);
    critical_point_sq_ = 4 * bandwidth_sq_ * (dims / 2.0 + 2) * math::LN_2;
    critical_point_value_ = EvalUnnormOnSq(critical_point_sq_);
  }
  
  /**
   * Evaluates an unnormalized density, given the distance between
   * the kernel's mean and a query point.
   */
  double EvalUnnorm(double dist) const {
    return EvalUnnormOnSq(dist * dist);
  }

  /**
   * Evaluates an unnormalized density, given the square of the
   * distance.
   */
  double EvalUnnormOnSq(double sqdist) const {
    double d =
      factor_ * exp(sqdist * neg_inv_bandwidth_2sq_ * 0.5)
      - exp(sqdist * neg_inv_bandwidth_2sq_);
    return d;
  }

  /** Unnormalized range on a range of squared distances. */
  DRange RangeUnnormOnSq(const DRange& range) const {
    double eval_lo = EvalUnnormOnSq(range.lo);
    double eval_hi = EvalUnnormOnSq(range.hi);
    if (range.lo < critical_point_sq_) {
      if (range.hi < critical_point_sq_) {
        // Strictly under critical point.
        return DRange(eval_lo, eval_hi);
      } else {
        // Critical point is included
        return DRange(std::min(eval_lo, eval_hi), critical_point_value_);
      }
    } else {
      return DRange(eval_hi, eval_lo);  //!! TODO explain
    }
  }

  /**
   * Divide by this constant when you're done.
   *
   * @deprecated -- this function is very confusing
   */
  double CalcNormConstant(index_t dims) const {
    return pow(math::PI_2*bandwidth_sq_, dims / 2) / 2;
  }
  
  /**
   * Multiply densities by this value.
   */
  double CalcMultiplicativeNormConstant(index_t dims) const {
    return 1.0 / CalcNormConstant(dims);
  }
};

/**
 * Multivariate Epanechnikov kernel.
 *
 * To use, first get an unnormalized density, and divide by the
 * normalizeation factor.
 */
class EpanKernel {
 private:
  double inv_bandwidth_sq_;
  double bandwidth_sq_;

  OBJECT_TRAVERSAL(EpanKernel) {
    OT_OBJ(inv_bandwidth_sq_);
    OT_OBJ(bandwidth_sq_);
  }
  
 public:
  static const bool HAS_CUTOFF = true;
  
 public:
  void Init(double bandwidth_in, index_t dims) {
    Init(bandwidth_in);
  }

  /**
   * Initializes to a specific bandwidth.
   */
  void Init(double bandwidth_in) {
    bandwidth_sq_ = (bandwidth_in * bandwidth_in);
    inv_bandwidth_sq_ = 1.0 / bandwidth_sq_;
  }
  
  /**
   * Evaluates an unnormalized density, given the distance between
   * the kernel's mean and a query point.
   */
  double EvalUnnorm(double dist) const {
    return EvalUnnormOnSq(dist * dist);
  }
  
  /**
   * Evaluates an unnormalized density, given the square of the
   * distance.
   */
  double EvalUnnormOnSq(double sqdist) const {
    // TODO: Try the fabs non-branching version.
    if (sqdist < bandwidth_sq_) {
      return 1 - sqdist * inv_bandwidth_sq_;
    } else {
      return 0;
    }
  }

  /** Unnormalized range on a range of squared distances. */
  DRange RangeUnnormOnSq(const DRange& range) const {
    return DRange(EvalUnnormOnSq(range.hi), EvalUnnormOnSq(range.lo));  //!! TODO explain
  }

  /**
   * Gets the maximum unnormalized value.
   */
  double MaxUnnormValue() {
    return 1.0;
  }
  
  /**
   * Divide by this constant when you're done.
   */
  double CalcNormConstant(index_t dims) const {
    return 2.0 * math::SphereVolume(sqrt(bandwidth_sq_), dims)
        / (dims + 2.0);
  }
  
  /**
   * Gets the squared bandwidth.
   */
  double bandwidth_sq() const {
    return bandwidth_sq_;
  }
  
  /**
  * Gets the reciproccal of the squared bandwidth.
   */
  double inv_bandwidth_sq() const {
    return inv_bandwidth_sq_;
  }
};


#endif
