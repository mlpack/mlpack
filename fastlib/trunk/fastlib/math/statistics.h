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
 * @file statistics.h
 *
 * Statistics utilities.
 */

#ifndef MATH_STATISTICS_H
#define MATH_STATISTICS_H

#include "../base/base.h"
#include "../la/matrix.h"
#include <math.h>

namespace math {
  /**
   * Computes the mean value of a vector.
   * Don't forget initializing V before using this function
   *
   * @param V the input vector
   * @return the mean value
   */
  double Mean(Vector V);

  /**
   * Computes the variance of a vector using "corrected two-pass algorithm".
   * See "Numerical Recipes in C" for reference.
   * Don't forget initializing V before using this function.
   *
   * @param V the input vector
   * @return the variance
   */
  double Var(Vector V);

  /**
   * Computes the standard deviation of a vector.
   * Don't forget initializing V before using this function.
   *
   * @param V the input vector
   * @return the standard deviation
   */
  double Std(Vector V);

  /**
   * Computes the sigmoid function of a real number x
   * Sigmoid(x) = 1/[1+exp(-x)]
   *
   * @param x the input real number
   * @return the sigmoid function value
   */
  double Sigmoid(double x);
};

#endif
