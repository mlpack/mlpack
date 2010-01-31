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
#ifndef FASTLIB_DISCRETE_DISTRIBUTION_H
#define FASTLIB_DISCRETE_DISTRIBUTION_H
#include "fastlib/fastlib.h"
class DiscreteDST {
  Vector p;
  Vector ACC_p;
 public:
  void Init(int N = 2);
  void generate(int* v);
  double get(int i) { return p[i]; }
  void set(const Vector& p_) { p.CopyValues(p_); }

  void start_accumulate() { ACC_p.SetZero(); }
  void accumulate(int i, double v) { ACC_p[i]+=v; }
  void end_accumulate();
};
#endif
