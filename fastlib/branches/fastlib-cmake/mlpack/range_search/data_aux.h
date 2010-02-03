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
#ifndef DATA_AUX_H
#define DATA_AUX_H

#include "fastlib/fastlib.h"

namespace data_aux {

  /**
   * Loads a matrix from a file.
   *
   * This supports any type the Dataset class supports with the
   * InitFromFile function: CSV and ARFF.
   *
   * @code
   * Matrix A;
   * data::Load("foo.csv", &A);
   * @endcode
   *
   * @param fname the file name to load
   * @param matrix a pointer to an uninitialized matrix to load
   */
  template<typename T>
  static success_t Load(const char *fname, GenMatrix<T> *matrix) {
    Matrix tmp_matrix;
    success_t result = data::Load(fname, &tmp_matrix);

    // Allocate the matrix that is to be returned and copy all
    // entries.
    matrix->StaticInit(tmp_matrix.n_rows(), tmp_matrix.n_cols());
    for(index_t c = 0; c < tmp_matrix.n_cols(); c++) {
      for(index_t r = 0; r < tmp_matrix.n_rows(); r++) {
	matrix->set(r, c, STATIC_CAST(T, tmp_matrix.get(r, c)));
      }
    }

    return result;
  }

  /**
   * Loads a matrix from a file.
   *
   * This supports any type the Dataset class supports with the
   * InitFromFile function: CSV and ARFF.
   *
   * @code
   * Matrix A;
   * data::Load("foo.csv", &A);
   * @endcode
   *
   * @param fname the file name to load
   * @param matrix a pointer to an uninitialized matrix to load
   */
  template<typename T>
  static success_t LoadTranspose(const char *fname, GenMatrix<T> *matrix) {
    Matrix tmp_matrix;
    success_t result = data::Load(fname, &tmp_matrix);

    // Allocate the matrix that is to be returned and copy all
    // entries.
    matrix->StaticInit(tmp_matrix.n_cols(), tmp_matrix.n_rows());
    for(index_t c = 0; c < tmp_matrix.n_cols(); c++) {
      for(index_t r = 0; r < tmp_matrix.n_rows(); r++) {
	matrix->set(c, r, STATIC_CAST(T, tmp_matrix.get(r, c)));
      }
    }

    return result;
  }

};

#endif
