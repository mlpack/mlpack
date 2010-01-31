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
#ifndef INVERSE_POW_DIST_KERNEL_H
#define INVERSE_POW_DIST_KERNEL_H

#include "fastlib/fastlib.h"

class InversePowDistGradientKernel {

 public:
  double lambda_;
  
  index_t dimension_;

 public:
  
  void Init(double lambda_in, index_t dimension_in) {
    lambda_ = lambda_in;
    dimension_ = dimension_in;
  }

  double EvalUnnorm(const double *point) const {
    double sqdist = la::Dot(dimension_, point, point);
    return point[dimension_] / pow(sqdist, lambda_ / 2.0);
  }
};

class InversePowDistKernel {
  
 public:
  double lambda_;
  
  index_t dimension_;

 public:

  void Init(double lambda_in, index_t dimension_in) {
    lambda_ = lambda_in;
    dimension_ = dimension_in;
  }

  double EvalUnnorm(const double *point) const {
    double sqdist = la::Dot(dimension_, point, point);

    if(lambda_ > 0) {
      return 1.0 / pow(sqdist, lambda_ / 2.0);
    }
    else {
      return pow(sqdist, -lambda_ / 2.0);
    }
  }

  double EvalUnnorm(double dist) const {
    return EvalUnnormOnSq(dist * dist);
  }

  double EvalUnnormOnSq(double sqdist) const {
    if(lambda_ > 0) {
      return 1.0 / pow(sqdist, lambda_ / 2.0);
    }
    else {
      return pow(sqdist, -lambda_ / 2.0);
    }
  }

  DRange RangeUnnormOnSq(const DRange &range) const {
    return DRange(EvalUnnormOnSq(range.hi), EvalUnnormOnSq(range.lo));
  }

};

#endif
