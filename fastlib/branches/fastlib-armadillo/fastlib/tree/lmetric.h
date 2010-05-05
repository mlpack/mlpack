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
 * @file tree/lmetric.h
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * This file describes the LMetric policy class.
 *
 * @experimental
 */

#include "../la/la.h"
#include "../la/matrix.h"

#include "../math/math_lib.h"

/**
 * An L_p metric for vector spaces.
 *
 * A generic Metric class should simply compute the distance between
 * two points.  An LMetric operates for integer powers on Vector spaces.
 */
template<int t_pow>
class LMetric {
  public:
    /**
     * Computes the distance metric between two points.
     */
    static double Distance(const Vector& a, const Vector& b) {
      return math::Pow<1, t_pow>(
          la::RawLMetric<t_pow>(a.length(), a.ptr(), b.ptr()));
    }

    /**
     * Computes the distance metric between two points, raised to a
     * particular power.
     *
     * This might be faster so that you could get, for instance, squared
     * L2 distance.
     */
    template<int t_result_pow>
      static double PowDistance(const Vector& a, const Vector& b) {
        return math::Pow<t_result_pow, t_pow>(
            la::RawLMetric<t_pow>(a.length(), a.ptr(), b.ptr()));
      }
};
