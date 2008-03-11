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
    matrix->Init(tmp_matrix.n_rows(), tmp_matrix.n_cols());
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
    matrix->Init(tmp_matrix.n_cols(), tmp_matrix.n_rows());
    for(index_t c = 0; c < tmp_matrix.n_cols(); c++) {
      for(index_t r = 0; r < tmp_matrix.n_rows(); r++) {
	matrix->set(c, r, STATIC_CAST(T, tmp_matrix.get(r, c)));
      }
    }

    return result;
  }

};

#endif
