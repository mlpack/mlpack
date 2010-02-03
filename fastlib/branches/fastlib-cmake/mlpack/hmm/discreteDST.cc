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
#include "fastlib/fastlib.h"
#include "discreteDST.h"
#include "support.h"

void DiscreteDST::Init(int N) {
  p.Init(N);
  ACC_p.Init(N);
  double s = 1;
  for (int i = 0; i < N-1; i++) {
    p[i] = RAND_UNIFORM(s*0.2,s*0.8);
    s -= p[i];
  }
  p[N-1] = s;
}

void DiscreteDST::generate(int* v) {
  int N = p.length();
  double r = RAND_UNIFORM_01;
  double s = 0;
  for (int i = 0; i < N; i++) {
    s += p[i];
    if (s >= r) {
      *v = i;
      return;
    }
  }
  *v = N-1;
}

void DiscreteDST::end_accumulate() {
  int N = p.length();
  double s = 0;
  for (int i = 0; i < N; i++) s += ACC_p[i];
  if (s == 0) s = -INFINITY;
  for (int i = 0; i < N; i++) p[i] = ACC_p[i]/s;
}
