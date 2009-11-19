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
 * @file statistics.cc
 *
 * Implementation for statistics helpers.
 */

#include "fastlib/math/statistics.h"
#include <math.h>
//#include "statistics.h"

namespace math {

double Mean(Vector V) {
  double c = 0.0;
  index_t n = V.length();
  for (index_t i=0; i<n; i++)
    c = c + V[i];
  return c / n;
}

double Var(Vector V) {
  double c = 0.0, mean, va, ep;
  index_t n = V.length();

  for (index_t i=0; i<n; i++)
    c = c + V[i];
  mean = c / n;

  ep = 0.0; va = 0.0;
  for (index_t i=0; i<n; i++) {
    c = V[i] - mean;
    ep = ep + c;
    va = va + c * c;
  }
  return (va - ep * ep / n) / (n - 1);
}

double Std(Vector V) {
  return sqrt( Var(V) );
}
  
double Sigmoid(double x) {
  return 1.0 / ( 1.0 + exp(-x) );
}
  

};
