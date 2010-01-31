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
 * @file sample.cc
 *
 * Useful for taking random samples of datasets.
 */

#include <fastlib/fastlib.h>

const char *help_text[] = {
  "sample - Randomizes a data set",
  "",
  "Parameters:",
  " --in file.txt       (input file)",
  " --out file100k.txt  (output file)",
  " --n 100000          (number of points; if omitted, entire dataset)",
  " --seed 31415926     (srand seed; if omitted, time is used)"
};

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  const char *in = fx_param_str_req(fx_root, "in");
  const char *out = fx_param_str_req(fx_root, "out");

  fx_timer_start(fx_root, "read_in_matrix");
  Matrix m_in;
  data::Load(in, &m_in);
  fx_timer_stop(fx_root, "read_in_matrix");

  index_t n = fx_param_int(fx_root, "n", m_in.n_cols());
  index_t seed = fx_param_int(fx_root, "seed", time(NULL));

  srand(seed);

  fx_timer_start(fx_root, "make_permutation");
  ArrayList<index_t> permutation;
  math::MakeRandomPermutation(m_in.n_cols(), &permutation);
  fx_timer_stop(fx_root, "make_permutation");

  fx_timer_start(fx_root, "make_new_matrix");
  Matrix m_out;
  m_out.Init(m_in.n_rows(), n);

  for (index_t i = 0; i < n; i++) {
    Vector v_in;
    Vector v_out;

    m_in.MakeColumnVector(permutation[i], &v_in);
    m_out.MakeColumnVector(i, &v_out);
    v_out.CopyValues(v_in);
  }
  fx_timer_stop(fx_root, "make_new_matrix");

  fx_timer_start(fx_root, "save_new_matrix");
  data::Save(out, m_out);
  fx_timer_stop(fx_root, "save_new_matrix");

  fx_done();
}

