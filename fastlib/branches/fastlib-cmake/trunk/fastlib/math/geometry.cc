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
 * @file geometry.cc
 *
 * Implementation for geometry helpers.
 */

#include "fastlib/math/geometry.h"
#include "fastlib/math/discrete.h"
#include "fastlib/math/math_lib.h"
//#include "geometry.h"
//#include "discrete.h"
//#include "math_lib.h"

namespace math {

  double SphereVolume(double r, int d) {
    int n = d / 2;
    double val;
    
    DEBUG_ASSERT(d >= 0);
    
    if (d % 2 == 0) {
      val = pow(r * sqrt(PI), d) / Factorial(n);
    }
    else {
      val = pow(2 * r, d) * pow(PI, n) * Factorial(n) / Factorial(d);
    }
    
    return val;
  }
  
};
