/** @file cyclic_dense_matrix.h
 *
 *  Represents a dense matrix with the starting column shifted
 *  cyclically.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_CYCLIC_DENSE_MATRIX_H
#define CORE_TABLE_CYCLIC_DENSE_MATRIX_H

#include <armadillo>
#include "core/table/dense_point.h"

namespace core {
namespace table {
class CyclicArmaMat {
  private:
    arma::mat mat_;

    int starting_column_index_;

  private:

    int translated_column_index_(int column_index) const {
      return (column_index + starting_column_index_) % mat_.n_cols;
    }

  public:

    int n_rows() const {
      return mat_.n_rows;
    }

    int n_cols() const {
      return mat_.n_cols;
    }

    void zeros(int row, int col) {
      mat_.zeros(row, col);
    }

    void ShiftStartingIndex(int shift = 1) {
      starting_column_index_ =
        (starting_column_index_ + shift) % mat_.n_cols;
    }

    void col(const int col_num, arma::vec *col_out) const {
      int trans_col_index = this->translated_column_index_(col_num);
      double *ptr =
        const_cast<double *>(mat_.memptr() + mat_.n_rows * trans_col_index);
      core::table::DoublePtrToArmaVec(ptr, mat_.n_rows, col_out);
    }

    double &at(int row, int col) {
      return mat_.at(row, this->translated_column_index_(col));
    }
};
}
}

#endif
